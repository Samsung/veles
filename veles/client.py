"""
Created on Jan 22, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
from copy import copy
import json
import numpy
import six
from six import add_metaclass
import time
from twisted.internet import reactor, threads
from twisted.internet.protocol import ReconnectingClientFactory
from twisted.python.failure import Failure
import zmq

from veles.cmdline import CommandLineArgumentsRegistry
import veles.error as error
import veles.external.fysom as fysom
from veles.external.txzmq import ZmqConnection, ZmqEndpoint, SharedIO
from veles.network_common import NetworkAgent, StringLineReceiver
from veles.thread_pool import errback


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
        try:
            receiver(self, *message[1:])
        except:
            errback(Failure())

    def request(self, command, message=b''):
        if not self.shmem is None and command == 'update':
            self.shmem.seek(0)
        try:
            pickles_size = self.send(
                self.id, command.encode('charmap'), message,
                io=self.shmem,
                pickles_compression=self.pickles_compression)
        except ZmqConnection.IOOverflow:
            self.shmem = None
            return
        if self.is_ipc and command == 'update' and self.shmem is None:
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
            {'name': 'request_job', 'src': ['WAIT', 'POSTPONED'],
                                    'dst': 'GETTING_JOB'},
            {'name': 'obtain_job', 'src': 'GETTING_JOB', 'dst': 'BUSY'},
            {'name': 'refuse_job', 'src': 'GETTING_JOB', 'dst': 'END'},
            {'name': 'postpone_job', 'src': 'GETTING_JOB', 'dst': 'POSTPONED'},
            {'name': 'complete_job', 'src': 'BUSY', 'dst': 'WAIT'},
        ],
        'callbacks': {
            'onchangestate': onFSMStateChanged
        }
    }

    def __init__(self, addr, factory, async=False, death_probability=0.0):
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
        self.death_probability = death_probability
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
        err = msg.get("error")
        if err:
            self.disconnect("Server returned error: '%s'", err)
            return
        if self.state.current == "WAIT":
            cid = msg.get("id")
            if not cid:
                self.factory.host.error("No ID was received in WAIT state")
                self.request_id()
                return
            self.factory.id = cid
            self.factory.host.info("My ID is %s", cid)
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
        elif job == b"NEED_UPDATE":
            self.factory.host.debug("Master returned NEED_UPDATE, will repeat "
                                    "the job request in "
                                    "update_result_received()")
            self.state.postpone_job()
        else:
            self.state.obtain_job()
        update = self._last_update
        if self.async and update is not None:
            self.request_update()
        if job == b"NEED_UPDATE":
            return
        if not job and not self.async:
            self.factory.host.launcher.stop()
            return
        try:
            if numpy.random.random() < self.death_probability:
                raise error.Bug("This slave has randomly crashed (death "
                                "probability was %f)" % self.death_probability)
            self.factory.host.workflow.do_job(job, update, self.job_finished)
        except:
            errback(Failure())

    def job_finished(self, update):
        if self.state.current != "BUSY":
            self.factory.host.error("Invalid state %s", self.state.current)
        self._last_update = update
        self.state.complete_job()
        if self.async:
            self.request_job()
        else:
            self.request_update()

    def update_result_received(self, result):
        if result == b'0':
            self.factory.host.warning("Last update was rejected")
        else:
            assert result == b'1'
            self.factory.host.debug("Update was confirmed")
        if self.state.current == "END":
            self.factory.host.launcher.stop()
            return
        if not self.async or self.state.current == "POSTPONED":
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

    def request_update(self):
        self.factory.host.debug("Sending the update...")
        update, self._last_update = self._last_update, None
        if self.async:
            # we have to copy the update since it may be overwritten in do_job
            update = copy(update)
        self.zmq_connection.request("update", update or b'')

    def disconnect(self, msg, *args, **kwargs):
        self.factory.host.error(msg, *args, **kwargs)
        self.state.disconnect()
        self.transport.loseConnection()


class VelesProtocolFactory(ReconnectingClientFactory):
    RECONNECTION_INTERVAL = 1
    RECONNECTION_ATTEMPTS = 60

    def __init__(self, host, async, death_probability):
        if six.PY3:
            super(VelesProtocolFactory, self).__init__()
        self.host = host
        self.id = None
        self.state = None
        self._async = async
        self._death_probability = death_probability
        self.disconnect_time = None

    def startedConnecting(self, connector):
        self.host.info('Connecting to %s:%s...',
                       self.host.address, self.host.port)

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self, self._async, self._death_probability)

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


@add_metaclass(CommandLineArgumentsRegistry)
class Client(NetworkAgent):
    """
    UDT/TCP client operating on a single socket.
    """

    def __init__(self, configuration, workflow):
        super(Client, self).__init__(configuration)
        self.workflow = workflow
        self.launcher = workflow.workflow
        parser = Client.init_parser()
        args, _ = parser.parse_known_args()
        self.factory = VelesProtocolFactory(self, args.async,
                                            args.death_probability)
        self._initial_data = None
        reactor.connectTCP(self.address, self.port, self.factory, timeout=300)

    @staticmethod
    def init_parser(**kwargs):
        """
        Initializes an instance of argparse.ArgumentParser.
        """
        parser = kwargs.get("parser", argparse.ArgumentParser())
        parser.add_argument("--async",
                            default=kwargs.get("async", False),
                            help="Activate asynchronous master-slave protocol "
                            "on slaves.", action='store_true')
        parser.add_argument("--death-probability", type=float, default=0.0,
                            help="Each slave will die with the probability "
                            "specified by this value.")
        return parser

    @property
    def initial_data(self):
        return self._initial_data

    def initialize(self):
        self._initial_data = self.workflow.generate_initial_data_for_master()
