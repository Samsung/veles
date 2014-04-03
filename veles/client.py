"""
Created on Jan 22, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import fysom
import json
import six
import time
from twisted.internet import reactor, threads
from twisted.internet.protocol import ReconnectingClientFactory
from txzmq import ZmqConnection, ZmqEndpoint
import zmq

from veles.network_common import NetworkAgent, StringLineReceiver


class ZmqDealer(ZmqConnection):
    socketType = zmq.constants.DEALER

    def __init__(self, nid, host, *endpoints):
        super(ZmqDealer, self).__init__(endpoints)
        self.id = nid.encode()
        self.host = host

    def messageReceived(self, message):
        if self.host.state.current == "GETTING_JOB":
            self.host.job_received(message[2])
        elif self.host.state.current == "WAIT":
            self.host.update_result_received(message[2])

    def request(self, command, message=b''):
        self.send([self.id, b''] + [command.encode(), message])


class VelesProtocol(StringLineReceiver):
    """A communication controller from client to server.

    Attributes:
        FSM_DESCRIPTION     The definition of the Finite State Machine of the
                            protocol.
    """

    def onFSMStateChanged(self, e):
        """
        Logs the current state transition.
        """
        self.factory.host.debug("state: %s, %s -> %s", e.event, e.src, e.dst)

    FSM_DESCRIPTION = {
        'initial': 'INIT',
        'events': [
            {'name': 'disconnect', 'src': '*', 'dst': 'ERROR'},
            {'name': 'reconnect', 'src': '*', 'dst': 'INIT'},
            {'name': 'request_id', 'src': ['INIT', 'WAIT'], 'dst': 'WAIT'},
            {'name': 'request_job', 'src': ['WAIT', 'GETTING_JOB'],
                                    'dst': 'GETTING_JOB'},
            {'name': 'obtain_job', 'src': 'GETTING_JOB', 'dst': 'BUSY'},
            {'name': 'refuse_job', 'src': 'GETTING_JOB', 'dst': 'END'},
            {'name': 'wait_update_notification', 'src': 'BUSY', 'dst': 'WAIT'},
        ],
        'callbacks': {
            'onchangestate': onFSMStateChanged
        }
    }

    def __init__(self, addr, factory):
        """
        Initializes the protocol.

        Parameters:
            addr        The address of the server (reported by Twisted).
            factory     The factory which produced this protocol.
        """
        super(VelesProtocol, self).__init__()
        self.addr = addr
        self.factory = factory
        self.id = 'None'
        if not factory.state:
            factory.state = fysom.Fysom(VelesProtocol.FSM_DESCRIPTION, self)
        self.state = factory.state

    def connectionMade(self):
        self.factory.host.info("Connected")
        self.factory.disconnect_time = None
        if self.state.current == "INIT" or self.state.current == "WAIT":
            self.request_id()
            return
        if self.state.current == "GETTING_JOB":
            self.send_id()
            self.request_job()
            return
        if self.state.current == "BUSY":
            self.send_id()
            self.state.obtain_job()
            return

    def connectionLost(self, reason):
        self.factory.host.debug("Connection was lost")

    def lineReceived(self, line):
        self.factory.host.debug("lineReceived %s:  %s", self.id, line)
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            self.factory.host.error("Could not parse the received line, "
                                    "dropping it")
            return
        error = msg.get("error")
        if error:
            self.disconnect("Server returned error: '%s'", error)
            return
        if self.state.current == "WAIT":
            cid = msg.get("id")
            if not cid:
                self.factory.host.error("No ID was received in WAIT state")
                self.request_id()
                return
            self.factory.id = cid
            endpoint = msg.get("endpoint")
            if not endpoint:
                self.factory.host.error("No endpoint was received")
                self.request_id()
                return
            self.zmq_connection = ZmqDealer(
                cid, self, ZmqEndpoint("connect", endpoint))
            self.factory.host.info("Connected to ZeroMQ endpoint %s",
                                   endpoint)
            self.request_job()
            return
        self.disconnect("Invalid state %s", self.state.current)

    def job_received(self, job):
        if job == bytes(False):
            self.factory.host.info("Job was refused")
            self.state.refuse_job()
            self.factory.host.launcher.stop()
            return
        djob = threads.deferToThreadPool(
            reactor,
            self.factory.host.workflow.thread_pool,
            self.factory.host.workflow.do_job,
            job)
        djob.addCallback(self.job_finished)
        self.state.obtain_job()

    def job_finished(self, update):
        if self.state.current == "BUSY":
            self.zmq_connection.request("update", update)
            self.state.wait_update_notification()
            return
        self.factory.host.error("Invalid state %s", self.state.current)

    def update_result_received(self, result):
        self.request_job()

    def sendLine(self, line):
        if six.PY3:
            super(VelesProtocol, self).sendLine(json.dumps(line))
        else:
            StringLineReceiver.sendLine(self, json.dumps(line))

    def _common_id(self):
        return {'power': self.factory.host.workflow.get_computing_power(),
                'checksum': self.factory.host.workflow.checksum(),
                'mid': self.factory.host.mid,
                'pid': self.factory.host.pid}

    def send_id(self):
        common = self._common_id()
        common['id'] = self.factory.id
        self.sendLine(common)

    def request_id(self):
        self.sendLine(self._common_id())
        self.state.request_id()

    def request_job(self):
        self.zmq_connection.request("job")
        self.state.request_job()

    def disconnect(self, msg, *args, **kwargs):
        self.factory.host.error(msg, *args, **kwargs)
        self.state.disconnect()
        self.transport.loseConnection()


class VelesProtocolFactory(ReconnectingClientFactory):
    RECONNECTION_INTERVAL = 1
    RECONNECTION_ATTEMPTS = 60

    def __init__(self, host):
        if six.PY3:
            super(VelesProtocolFactory, self).__init__()
        self.host = host
        self.id = None
        self.state = None
        self.disconnect_time = None

    def startedConnecting(self, connector):
        self.host.info('Connecting to %s:%s...',
                       self.host.address, self.host.port)

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self)

    def clientConnectionLost(self, connector, reason):
        if not self.state or self.state.current not in ['ERROR', 'END']:
            lost_state = "<None>"
            if self.state:
                lost_state = self.state.current
                self.state.reconnect()
            if not self.disconnect_time:
                self.disconnect_time = time.time()
            if ((time.time() - self.disconnect_time) //
                    VelesProtocolFactory.RECONNECTION_INTERVAL >
                    VelesProtocolFactory.RECONNECTION_ATTEMPTS):
                self.host.error("Max reconnection attempts reached, exiting")
                self.host.launcher.stop()
                return
            self.host.warning("Disconnected in %s state, trying to "
                              "reconnect...", lost_state)
            reactor.callLater(VelesProtocolFactory.RECONNECTION_INTERVAL,
                              connector.connect)
        else:
            self.host.info("Disconnected")
            if self.state.current == 'ERROR':
                self.host.launcher.stop()

    def clientConnectionFailed(self, connector, reason):
        self.host.warning('Connection failed. Reason: %s', reason)
        self.clientConnectionLost(connector, reason)


class Client(NetworkAgent):
    """
    UDT/TCP client operating on a single socket.
    """

    def __init__(self, configuration, workflow):
        super(Client, self).__init__(configuration)
        self.workflow = workflow
        self.launcher = workflow.workflow
        self.factory = VelesProtocolFactory(self)
        reactor.connectTCP(self.address, self.port, self.factory, timeout=300)
