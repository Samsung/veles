"""
Created on Jan 22, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
from copy import copy
import datetime
import json
import six
from six import add_metaclass
import time
from twisted.internet import reactor, threads
from twisted.internet.defer import CancelledError
from twisted.internet.protocol import ReconnectingClientFactory
from twisted.python.failure import Failure
import zmq

from veles.cmdline import CommandLineArgumentsRegistry
from veles.config import root
import veles.error as error
import veles.external.fysom as fysom
from veles.external.prettytable import PrettyTable
from veles.external.txzmq import ZmqConnection, ZmqEndpoint, SharedIO
from veles.network_common import NetworkAgent, StringLineReceiver, IDLogger
from veles.prng import get as get_rg
from veles.thread_pool import errback
from veles.timeit import timeit


class ZmqDealer(ZmqConnection):
    socketType = zmq.DEALER

    RECEIVERS = {
        'job':
        lambda self, message: self.host.job_received(message),
        'update':
        lambda self, message: self.host.update_result_received(message),
        'error':
        lambda self, message: self.host.disconnect(message)
    }

    RESERVE_SHMEM_SIZE = 0.05

    def __init__(self, nid, host, endpoint):
        super(ZmqDealer, self).__init__((endpoint,))
        self.id = nid.encode('charmap')
        self.host = host
        self.is_ipc = endpoint.address.startswith('ipc://')
        self.shmem = None
        self.pickles_compression = root.common.network_compression \
            if not self.is_ipc else None
        self._request_timings = {}
        self._command = None
        self._command_str = None
        self._receive_timing = (0.0, 0)

    def parseHeader(self, header):
        self._command_str = header[0].decode('charmap')
        self._command = ZmqDealer.RECEIVERS.get(self._command_str)

    def messageReceived(self, message):
        if self._command is None and self._command_str is None:
            self.parseHeader(message)
            event_type = "single"
        else:
            event_type = "end"
        if self._command is None:
            raise RuntimeError("Received an unknown command %s" %
                               self._command_str)
        try:
            self._command(self, *message[1:])
        except:
            errback(Failure())
        finally:
            self.event("ZeroMQ", event_type, dir="receive",
                       command=self._command_str, height=0.5)
            self._command = None
            self._command_str = None
            self._receive_timing = (
                self._receive_timing[0] + self.last_read_time,
                self._receive_timing[1] + 1)

    def messageHeaderReceived(self, header):
        self.parseHeader(header)
        self.event("ZeroMQ", "begin", dir="receive", command=self._command_str,
                   height=0.5)

    @property
    def request_timings(self):
        return {key: val[0] / (val[1] or 1)
                for key, val in self._request_timings.items()}

    @property
    def receive_timing(self):
        return self._receive_timing[0] / (self._receive_timing[1] or 1)

    @property
    def total_receive_time(self):
        return self._receive_timing[0]

    @property
    def total_request_time(self):
        return sum((val[0] for val in self._request_timings.values()))

    def request(self, command, message=b''):
        self.event("ZeroMQ", "begin", dir="send", command=command, height=0.5)
        if self.shmem is not None and command == 'update':
            self.shmem.seek(0)
        try:
            pickles_size, delta = timeit(
                self.send,
                self.id, command.encode('charmap'), message,
                io=self.shmem,
                pickles_compression=self.pickles_compression)
            if command not in self._request_timings:
                self._request_timings[command] = (0.0, 0)
            self._request_timings[command] = (
                self._request_timings[command][0] + delta,
                self._request_timings[command][1] + 1)
        except ZmqConnection.IOOverflow:
            self.shmem = None
            return
        if self.is_ipc and command == 'update' and self.shmem is None:
            self.shmem = SharedIO(
                "veles-update-" + self.id.decode('charmap'),
                int(pickles_size * (1.0 + ZmqDealer.RESERVE_SHMEM_SIZE)))
        self.event("ZeroMQ", "end", dir="send", command=command, height=0.5)


class VelesProtocol(StringLineReceiver, IDLogger):
    """A communication controller from client to server.

    Attributes:
        FSM_DESCRIPTION     The definition of the Finite State Machine of the
                            protocol.
    """

    def onFSMStateChanged(self, e):
        """
        Logs the current state transition.
        """
        self.debug("state: %s, %s -> %s", e.event, e.src, e.dst)

    FSM_DESCRIPTION = {
        'initial': 'INIT',
        'events': [
            {'name': 'disconnect', 'src': '*', 'dst': 'ERROR'},
            {'name': 'reconnect', 'src': '*', 'dst': 'INIT'},
            {'name': 'request_id', 'src': ['INIT', 'WAIT'], 'dst': 'WAIT'},
            {'name': 'send_id', 'src': 'INIT', 'dst': 'WAIT'},
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

    def __init__(self, addr, host):
        """
        Initializes the protocol.

        Parameters:
            addr        The address of the server (reported by Twisted).
            factory     The factory which produced this protocol.
        """
        super(VelesProtocol, self).__init__(logger=host.logger)
        self.addr = addr
        self.host = host
        self._last_update = None
        self.state = host.state
        self._current_deferred = None
        self._power_upload_time = 0
        self._power_upload_threshold = 60
        self.rand = get_rg()

    def connectionMade(self):
        self.info("Connected in %s state", self.state.current)
        self.disconnect_time = None
        if self.id is None:
            self.request_id()
            return
        self.send_id()
        self.state.send_id()

    def connectionLost(self, reason):
        self.debug("Connection was lost")
        if self._current_deferred is not None:
            self._current_deferred.cancel()

    def lineReceived(self, line):
        self.debug("lineReceived:  %s", line)
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            self.error("Could not parse the received line, dropping it")
            return
        err = msg.get("error")
        if err:
            self.disconnect("Server returned error: '%s'", err)
            return
        if self.state.current == "WAIT":
            if msg.get("reconnect") == "ok":
                if self.id is None:
                    self.error("Server returned a successful reconnection, "
                               "but my ID is None")
                    self.request_id()
                    return
                self.request_job()
                return
            cid = msg.get("id")
            if cid is None:
                self.error("No ID was received in WAIT state")
                self.request_id()
                return
            self.id = cid
            self.debug("Received ID")
            log_id = msg.get("log_id")
            if log_id is None:
                self.error("No log ID was received in WAIT state")
                self.request_id()
                return
            self.host.on_id_received(self.id, log_id)
            endpoint = msg.get("endpoint")
            if endpoint is None:
                self.error("No endpoint was received")
                self.request_id()
                return
            self.host.zmq_connection = self.zmq_connection = ZmqDealer(
                cid, self, ZmqEndpoint("connect", endpoint))
            self.info("Connected to ZeroMQ endpoint %s", endpoint)
            data = msg.get('data')
            if data is not None:
                self._set_deferred(
                    self.host.workflow.apply_initial_data_from_master,
                    data)
            self.request_job()
            return
        self.disconnect("Invalid state %s", self.state.current)

    def job_received(self, job):
        if not job:
            self.info("Job was refused")
            self.state.refuse_job()
        elif job == b"NEED_UPDATE":
            self.debug("Master returned NEED_UPDATE, will repeat the job "
                       "request in update_result_received()")
            self.state.postpone_job()
        else:
            self.state.obtain_job()
        update = self._last_update
        if self.host.async and update is not None:
            self.request_update()
        if job == b"NEED_UPDATE":
            return
        if not job and not self.host.async:
            self.host.launcher.stop()
            return
        try:
            if self.rand.random() < self.host.death_probability:
                raise error.Bug("This slave has randomly crashed (death "
                                "probability was %f)" %
                                self.host.death_probability)
            now = time.time()
            if now - self._power_upload_time > self._power_upload_threshold:
                self._power_upload_time = now
                self.sendLine({
                    'cmd': 'change_power',
                    'power': self.host.workflow.computing_power})
            # workflow.do_job may hang, so launch it in the thread pool
            self._set_deferred(self.host.workflow.do_job, job, update,
                               self.job_finished)
        except:
            errback(Failure())

    def _set_deferred(self, f, *args, **kwargs):
        self._current_deferred = threads.deferToThreadPool(
            reactor, self.host.workflow.thread_pool,
            f, *args, **kwargs)

        def cancellable_errback(err):
            if err.type == CancelledError:
                return
            errback(err)

        self._current_deferred.addErrback(cancellable_errback)
        return self._current_deferred

    def job_finished(self, update):
        if self.state.current != "BUSY":
            self.error("Invalid state %s", self.state.current)
            return
        self._last_update = update
        self.state.complete_job()
        if self.host.async:
            self.request_job()
        else:
            self.request_update()

    def update_result_received(self, result):
        if result == b'0':
            self.warning("Last update was rejected")
        else:
            assert result == b'1'
            self.debug("Update was confirmed")
        if self.state.current == "END":
            self.host.launcher.stop()
            return
        if not self.host.async or self.state.current == "POSTPONED":
            self.request_job()

    def sendLine(self, line):
        if six.PY3:
            super(VelesProtocol, self).sendLine(json.dumps(line))
        else:
            StringLineReceiver.sendLine(self, json.dumps(line))

    def _common_id(self):
        return {'cmd': 'handshake',
                'power': self.host.workflow.computing_power,
                'checksum': self.host.workflow.checksum,
                'mid': self.host.mid,
                'pid': self.host.pid}

    def send_id(self):
        common = self._common_id()
        common['id'] = self.id
        self.sendLine(common)

    def request_id(self):
        request = self._common_id()
        request['data'] = self.host.initial_data
        self.sendLine(request)
        self.state.request_id()

    def request_job(self):
        self.state.request_job()
        self.zmq_connection.request("job")

    def request_update(self):
        self.debug("Sending the update...")
        update, self._last_update = self._last_update, None
        if self.host.async:
            # we have to copy the update since it may be overwritten in do_job
            update = copy(update)
        self.zmq_connection.request("update", update or b'')

    def disconnect(self, msg, *args, **kwargs):
        self.error(msg, *args, **kwargs)
        self.state.disconnect()
        self.transport.loseConnection()


@add_metaclass(CommandLineArgumentsRegistry)
class Client(NetworkAgent, ReconnectingClientFactory):
    """
    Twisted factory which operates on a TCP socket for commands and a ZeroMQ
    endpoint for data exchange.
    """

    def __init__(self, configuration, workflow, timeout=300,
                 reconnection_interval=1, reconnection_attempts=60):
        super(Client, self).__init__(configuration, workflow)
        parser = Client.init_parser()
        args, _ = parser.parse_known_args()
        self._async = args.async
        self._death_probability = args.death_probability
        self._initial_data = None
        self.id = None
        self.state = fysom.Fysom(VelesProtocol.FSM_DESCRIPTION, self)
        self.zmq_connection = None
        self.disconnect_time = None
        self.reconnection_interval = 1
        self.reconnection_attempts = reconnection_attempts
        reactor.connectTCP(self.address, self.port, self, timeout=timeout)

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

    @property
    def async(self):
        return self._async

    @property
    def death_probability(self):
        return self._death_probability

    def initialize(self):
        self._initial_data = self.workflow.generate_initial_data_for_master()

    def print_stats(self):
        if self.zmq_connection is None:
            return
        table = PrettyTable("", "receive", "send")
        table.align[""] = "r"
        table.add_row(
            "all",
            datetime.timedelta(seconds=self.zmq_connection.total_receive_time),
            datetime.timedelta(seconds=self.zmq_connection.total_request_time))
        try:
            table.add_row(
                "avg",
                datetime.timedelta(seconds=self.zmq_connection.receive_timing),
                datetime.timedelta(seconds=self.zmq_connection.request_timings[
                    "update"]))
        except KeyError:
            pass
        self.info("Timings:\n%s", table)

    def startedConnecting(self, connector):
        self.info('Connecting to %s:%s...', self.address, self.port)

    def buildProtocol(self, addr):
        self.protocol = VelesProtocol(addr, self)
        self.state.owner = self.protocol
        return self.protocol

    def clientConnectionLost(self, connector, reason):
        if self.state is None or self.state.current not in ['ERROR', 'END']:
            lost_state = "<None>"
            if self.state is not None:
                lost_state = self.state.current
                self.state.reconnect()
            if self.disconnect_time is None:
                self.disconnect_time = time.time()
            if ((time.time() - self.disconnect_time) //
                    self.reconnection_interval > self.reconnection_attempts):
                self.error("Max reconnection attempts reached, exiting")
                self.launcher.stop()
                return
            self.warning("Disconnected in %s state, trying to reconnect...",
                         lost_state)
            reactor.callLater(self.reconnection_interval, connector.connect)
        else:
            self.info("Disconnected")
            if self.state.current == 'ERROR':
                self.launcher.stop()

    def clientConnectionFailed(self, connector, reason):
        self.warning('Connection failed. Reason: %s', reason)
        self.clientConnectionLost(connector, reason)

    def on_id_received(self, node_id, log_id):
        pass
