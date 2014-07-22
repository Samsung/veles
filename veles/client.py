"""
Created on Jan 22, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import veles.external.fysom as fysom
import json
import six
import time
from twisted.internet import reactor, threads
from twisted.internet.protocol import ReconnectingClientFactory
from veles.external.txzmq import ZmqConnection, ZmqEndpoint, SharedIO
import zmq

from veles.network_common import NetworkAgent, StringLineReceiver


def errback(failure):
    reactor.callFromThread(failure.raiseException)


class ZmqDealer(ZmqConnection):
    socketType = zmq.DEALER

    RECEIVERS = {
        b'job':
        lambda self, message: self.host.job_received(message),
        b'update':
        lambda self, message: self.host.update_result_received(message),
        b'error':
        lambda self, message: self.host.disconnect(message)
    }

    RESERVE_SHMEM_SIZE = 0.05

    def __init__(self, nid, host, endpoint):
        super(ZmqDealer, self).__init__((endpoint,))
        self.id = nid.encode('charmap')
        self.host = host
        self.is_ipc = endpoint.address.startswith('ipc://')
        self.shmem = None
        self.pickles_compression = "snappy" if not self.is_ipc else None

    def messageReceived(self, message):
        command = message[0]
        receiver = ZmqDealer.RECEIVERS.get(command)
        if receiver is None:
            raise RuntimeError("Received an unknown command %s" %
                               str(command))
        receiver(self, *message[1:])

    def request(self, command, message=b''):
        if not self.shmem is None and command == 'update':
            self.shmem.seek(0)
        try:
            pickles_size = self.send(
                self.id, command.encode('charmap'), message,
                io=self.shmem,
                pickles_compression=self.pickles_compression)
            io_overflow = False
        except ZmqConnection.IOOverflow:
            self.shmem = None
            io_overflow = True
            return
        if self.is_ipc and command == 'update':
            if io_overflow or self.shmem is None:
                self.shmem = SharedIO(
                    "veles-update-" + self.id.decode('charmap'),
                    int(pickles_size * (1.0 + ZmqDealer.RESERVE_SHMEM_SIZE)))


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
            {'name': 'complete_job', 'src': 'BUSY', 'dst': 'WAIT'},
        ],
        'callbacks': {
            'onchangestate': onFSMStateChanged
        }
    }

    def __init__(self, addr, factory, async=False):
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
        self._last_update = None
        self.async = async
        if not factory.state:
            factory.state = fysom.Fysom(VelesProtocol.FSM_DESCRIPTION, self)
        self.state = factory.state

    @property
    def async(self):
        return self._async

    @async.setter
    def async(self, value):
        self._async = value

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
            data = msg.get('data')
            if data is not None:
                threads.deferToThreadPool(
                    reactor, self.factory.host.workflow.thread_pool,
                    self.factory.host.workflow.apply_initial_data_from_master,
                    data).addErrback(errback)
            self.request_job()
            return
        self.disconnect("Invalid state %s", self.state.current)

    def job_received(self, job):
        if not job:
            self.factory.host.info("Job was refused")
            self.state.refuse_job()
            self.factory.host.launcher.stop()
            return
        self.state.obtain_job()
        try:
            self.factory.host.workflow.do_job(job, self._last_update,
                                              self.job_finished)
        except:
            from twisted.python.failure import Failure
            errback(Failure())

    def job_finished(self, update):
        if self.state.current == "BUSY":
            self._last_update = update
            self.state.complete_job()
            if self.async:
                self.request_job()
            self.zmq_connection.request("update", update or b'')
            return
        self.factory.host.error("Invalid state %s", self.state.current)

    def update_result_received(self, result):
        if result is False:
            self.factory.host.warning("Last update was rejected")
        self.factory.host.debug("Update was confirmed")
        if not self.async:
            self.request_job()

    def sendLine(self, line):
        if six.PY3:
            super(VelesProtocol, self).sendLine(json.dumps(line))
        else:
            StringLineReceiver.sendLine(self, json.dumps(line))

    def _common_id(self):
        return {'power': self.factory.host.workflow.computing_power,
                'checksum': self.factory.host.workflow.checksum(),
                'mid': self.factory.host.mid,
                'pid': self.factory.host.pid}

    def send_id(self):
        common = self._common_id()
        common['id'] = self.factory.id
        self.sendLine(common)

    def request_id(self):
        request = self._common_id()
        request['data'] = self.factory.host.initial_data
        self.sendLine(request)
        self.state.request_id()

    def request_job(self):
        self.state.request_job()
        self.zmq_connection.request("job")

    def disconnect(self, msg, *args, **kwargs):
        self.factory.host.error(msg, *args, **kwargs)
        self.state.disconnect()
        self.transport.loseConnection()


class VelesProtocolFactory(ReconnectingClientFactory):
    RECONNECTION_INTERVAL = 1
    RECONNECTION_ATTEMPTS = 60

    def __init__(self, host, async):
        if six.PY3:
            super(VelesProtocolFactory, self).__init__()
        self.host = host
        self.id = None
        self.state = None
        self._async = async
        self.disconnect_time = None

    def startedConnecting(self, connector):
        self.host.info('Connecting to %s:%s...',
                       self.host.address, self.host.port)

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self, self._async)

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

    def __init__(self, configuration, workflow, async):
        super(Client, self).__init__(configuration)
        self.workflow = workflow
        self.launcher = workflow.workflow
        self.factory = VelesProtocolFactory(self, async)
        self._initial_data = None
        reactor.connectTCP(self.address, self.port, self.factory, timeout=300)

    @property
    def initial_data(self):
        return self._initial_data

    def initialize(self):
        self._initial_data = self.workflow.generate_initial_data_for_master()
