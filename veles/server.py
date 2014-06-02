"""
Created on Jan 14, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from collections import namedtuple
import json
import six
import socket
from twisted.internet import reactor, threads
from twisted.internet.protocol import ServerFactory
import uuid
import zmq

import veles.external.fysom as fysom
from veles.external.txzmq import ZmqConnection, ZmqEndpoint
from veles.network_common import NetworkAgent, StringLineReceiver


class ZmqRouter(ZmqConnection):
    socketType = zmq.ROUTER

    COMMANDS = {
        b'job':
        lambda protocol, payload: protocol.jobRequestReceived(),
        b'update':
        lambda protocol, payload: protocol.updateReceived(payload)
    }

    def __init__(self, host, *endpoints, **kwargs):
        super(ZmqRouter, self).__init__(endpoints)
        ignore_unknown_commands = kwargs.get("ignore_unknown_commands", False)
        self.host = host
        self.routing = {}
        self.ignore_unknown_commands = ignore_unknown_commands

    def messageReceived(self, message):
        routing, node_id, *payload = message
        node_id = node_id.decode('charmap')
        self.routing[node_id] = routing
        protocol = self.host.factory.protocols.get(node_id)
        if protocol is None:
            self.host.error("ZeroMQ sent unknown node ID %s", node_id)
            self.reply(node_id, False)
            return
        if len(payload) != 2:
            self.host.error("ZeroMQ sent an invalid payload %s with node ID "
                            "%s", str(payload), node_id)
            self.reply(node_id, False)
            return
        command = ZmqRouter.COMMANDS.get(payload[0])
        if command is None and not self.ignore_unknown_commands:
            self.host.error("Received an unknown command %s with node ID %s",
                            command.decode(), node_id)
            self.reply(node_id, False)
            return
        command(protocol, payload[1])

    def reply(self, node_id, message):
        self.send(self.routing.pop(node_id), True)
        self.send_pickled(message)


SlaveDescription = namedtuple("SlaveDescription",
                              ['id', 'mid', 'pid', 'power', 'host', 'state'])


def make_slave_desc(info):
    args = dict(info)
    for f in SlaveDescription._fields:
        if f not in args:
            args[f] = None
    for f in info:
        if f not in SlaveDescription._fields:
            del args[f]
    return SlaveDescription(**args)


class VelesProtocol(StringLineReceiver):
    """A communication controller from server to client.

    Attributes:
        FSM_DESCRIPTION     The definition of the Finite State Machine of the
                            protocol.
    """

    def onFSMStateChanged(self, e):
        """
        Logs the current state transition.
        """
        self.host.debug("%s state: %s, %s -> %s",
                        self.id, e.event, e.src, e.dst)

    def onConnected(self, e):
        self.host.info("Accepted %s", self.address)

    def onIdentified(self, e):
        self.host.info("New node %s joined (%s)",
                       self.id, str(self.nodes[self.id]))

    def onJobObtained(self, e):
        self.nodes[self.id]["state"] = "Working"

    def setWaiting(self, e):
        self.nodes[self.id]["state"] = "Waiting"

    def onDropped(self, e):
        self.host.warning("Lost connection with %s", self.id or self.address)
        if self.id in self.nodes:
            self.nodes[self.id]["state"] = "Offline"

    FSM_DESCRIPTION = {
        'initial': 'INIT',
        'events': [
            {'name': 'connect', 'src': 'INIT', 'dst': 'WAIT'},
            {'name': 'identify', 'src': 'WAIT', 'dst': 'WORK'},
            {'name': 'receive_update', 'src': 'WORK',
                                       'dst': 'APPLYING_UPDATE'},
            {'name': 'apply_update', 'src': 'APPLYING_UPDATE', 'dst': 'WORK'},
            {'name': 'request_job', 'src': 'WORK', 'dst': 'GETTING_JOB'},
            {'name': 'obtain_job', 'src': 'GETTING_JOB', 'dst': 'WORK'},
            {'name': 'refuse_job', 'src': 'GETTING_JOB', 'dst': 'WORK'},
            {'name': 'drop', 'src': '*', 'dst': 'INIT'},
        ],
        'callbacks': {
            'onchangestate': onFSMStateChanged,
            'onconnect': onConnected,
            'onidentify': onIdentified,
            'onobtain_job': onJobObtained,
            'onreceive_update': setWaiting,
            'onWORK': setWaiting,
            'ondrop': onDropped
        }
    }

    def __init__(self, addr, factory):
        """
        Initializes the protocol.

        Parameters:
            addr    The address of the client (reported by Twisted).
            nodes   The clients which are known (dictionary, the key is ID).
            factory An instance of producing VelesProtocolFactory.
        """
        super(VelesProtocol, self).__init__()
        self.addr = addr
        self.factory = factory
        self.host = factory.host
        self.nodes = self.host.nodes
        self._id = None
        self._not_a_slave = False
        self.state = fysom.Fysom(VelesProtocol.FSM_DESCRIPTION, self)
        self._responders = {"WAIT": self._respondInWAIT,
                            "WORK": self._respondInWORK}

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if self._id is not None:
            del self.factory.protocols[self._id]
        self._id = value
        self.factory.protocols[self._id] = self

    @property
    def address(self):
        return "%s:%d" % (self.addr.host, self.addr.port)

    @property
    def not_a_slave(self):
        return self._not_a_slave

    def connectionMade(self):
        self.hip = self.transport.getHost().host
        self.state.connect()

    def connectionLost(self, reason):
        if self.not_a_slave:
            return
        self.state.drop()
        if not self.host.workflow.is_running:
            if self.id in self.nodes:
                del self.nodes[self.id]
            del self.factory.protocols[self.id]
            if len(self.nodes) == 0:
                self.host.launcher.stop()
        elif self.id in self.nodes:
            threads.deferToThreadPool(
                reactor, self.host.workflow.thread_pool,
                self.host.workflow.drop_slave,
                make_slave_desc(self.nodes[self.id]))
            if self.id in self.factory.protocols:
                del self.factory.protocols[self.id]

    def lineReceived(self, line):
        self.host.debug("%s lineReceived:  %s", self.id, line)
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            self.host.error("%s Could not parse the received line, dropping "
                            "it", self.id)
            return
        if self._checkQuery(msg):
            return
        if self.not_a_slave:
            self._sendError("You must reconnect as a slave to send commands")
        responder = self._responders.get(self.state.current)
        if responder is not None:
            responder(msg, line)
        else:
            self._sendError("You sent me something which is not allowed in my "
                            "current state %s" % self.state.current)

    def jobRequestReceived(self):
        self.state.request_job()
        self._requestJob()

    def jobRequestFinished(self, data):
        if self.state.current != "GETTING_JOB":
            return
        if data is not None:
            if not data:
                # Try again later
                self._requestJob()
                return
            self.state.obtain_job()
            self.host.debug("%s Job size: %d Kb", self.id, len(data) / 1000)
            self.host.zmq_connection.reply(self.id, data)
        else:
            self.state.refuse_job()
            self.host.zmq_connection.reply(self.id, False)

    def updateReceived(self, data):
        self.state.receive_update()
        upd = threads.deferToThreadPool(
            reactor, self.host.workflow.thread_pool,
            self.host.workflow.apply_update, data,
            make_slave_desc(self.nodes[self.id]))
        upd.addCallback(self.updateFinished)

    def updateFinished(self, result):
        if self.state.current != 'APPLYING_UPDATE':
            # silently ignore anything received not in APPLYING_UPDATE state
            return
        self.state.apply_update()
        if result:
            self.host.zmq_connection.reply(self.id, True)
        else:
            self.host.zmq_connection.reply(self.id, False)

    def sendLine(self, line):
        if six.PY3:
            super(VelesProtocol, self).sendLine(json.dumps(line))
        else:
            StringLineReceiver.sendLine(self, json.dumps(line))

    def _checkQuery(self, msg):
        """Respond to possible informational requests.
        """
        query = msg.get('query')
        if query is None:
            return False
        self._not_a_slave = True
        responders = {"nodes": lambda _: self.sendLine(self.nodes),
                      "endpoints":
                      lambda _: self.sendLine(self.host.zmq_endpoints)}
        responder = responders.get(query)
        if responder is None:
            self._sendError("%s query is not supported" % query)
        else:
            responder()

    def _respondInWAIT(self, msg, line):
        mysha = self.host.workflow.checksum()
        your_sha = msg.get("checksum")
        if not your_sha:
            self.host.error("Did not receive the workflow checksum")
            self._sendError("Workflow checksum is missing")
            return
        if mysha != your_sha:
            self._sendError("Workflow checksum mismatch")
            return
        must_reply = False
        msgid = msg.get("id")
        if not msgid:
            self.id = str(uuid.uuid4())
            must_reply = True
        else:
            self.id = msgid
            if not self.nodes.get(self.id):
                self.host.warning("Did not recognize the received ID %s",
                                  self.id)
                must_reply = True
        if must_reply:
            try:
                _, mid, pid = self._extractClientInformation(msg)
            except Exception as e:
                self.host.error(str(e))
                return
            data = self.host.workflow.generate_initial_data_for_slave(
                make_slave_desc(self.nodes[self.id]))
            retmsg = {'endpoint': self.host.endpoint(mid, pid, self.hip),
                      'data': data}
            if not msgid:
                retmsg['id'] = self.id
            self.sendLine(retmsg)
        data = msg.get('data')
        if data is not None:
            threads.deferToThreadPool(
                reactor, self.host.workflow.thread_pool,
                self.host.workflow.apply_initial_data_from_slave,
                data, make_slave_desc(self.nodes[self.id]))
            self.nodes[self.id]['data'] = [d for d in data if d is not None]
        self.state.identify()

    def _respondInWORK(self, msg, line):
        cmd = msg.get("cmd")
        if not cmd:
            self.host.error("%s Client sent something which is not a command: "
                            "%s. Sending back the error message",
                            self.id, line)
            self._sendError("No command found")
            return
        self.host.error("%s Unsupported %s command. Sending back the error "
                        "message", self.id, cmd)
        self._sendError("Unsupported command")

    def _extractClientInformation(self, msg):
        power = msg.get("power")
        mid = msg.get("mid")
        pid = msg.get("pid")
        if power is None:
            self._sendError("I need your computing power")
            raise Exception("Newly connected client did not send "
                            "it's computing power value, sending back "
                            "the error message")
        if mid is None:
            self._sendError("I need your machine id")
            raise Exception("Newly connected client did not send "
                            "it's machine id, sending back the error "
                            "message")
        if pid is None:
            self._sendError("I need your process id")
            raise Exception("Newly connected client did not send "
                            "it's process id, sending back the error "
                            "message")
        self.nodes[self.id] = {'power': power, 'mid': mid, 'pid': pid,
                               'id': self.id}
        reactor.callLater(0, self._resolveAddr, self.addr)
        return power, mid, pid

    def _resolveAddr(self, addr):
        host, _, _ = socket.gethostbyaddr(addr.host)
        if host == "localhost":
            host = socket.gethostname()
        self.host.debug("%s Address %s was resolved to %s", self.id,
                        addr.host, host)
        self.nodes[self.id]['host'] = host

    def _sendError(self, err):
        """
        Sends the line with the specified error message.

        Parameters:
            err:    The error message.
        """
        self.host.error(err)
        self.sendLine({'error': err})

    def _requestJob(self):
        job = threads.deferToThreadPool(
            reactor, self.host.workflow.thread_pool,
            self.host.workflow.generate_data_for_slave,
            make_slave_desc(self.nodes[self.id]))
        job.addCallback(self.jobRequestFinished)


class VelesProtocolFactory(ServerFactory):
    def __init__(self, host):
        if six.PY3:
            super(VelesProtocolFactory, self).__init__()
        self.host = host
        self.protocols = {}

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self)


class Server(NetworkAgent):
    """
    UDT/TCP server operating on a single socket
    """

    def __init__(self, configuration, workflow):
        super(Server, self).__init__(configuration)
        self.nodes = {}
        self.workflow = workflow
        self.launcher = workflow.workflow
        self.factory = VelesProtocolFactory(self)
        reactor.listenTCP(self.port, self.factory, interface=self.address)
        self.info("Accepting new connections on %s:%d",
                  self.address, self.port)
        try:
            self.zmq_connection = ZmqRouter(
                self, ZmqEndpoint("bind", "inproc://veles"),
                ZmqEndpoint("bind", "rndipc://veles-ipc-:"),
                ZmqEndpoint("bind", "rndtcp://*:1024:65535:1"))
        except zmq.error.ZMQBindError:
            self.exception("Could not setup ZeroMQ socket")
            raise
        self.zmq_ipc_fn, self.zmq_tcp_port = self.zmq_connection.rnd_vals
        self.zmq_endpoints = {"inproc": "inproc://veles",
                              "ipc": "ipc://%s" % self.zmq_ipc_fn,
                              "tcp": "tcp://*:%d" % self.zmq_tcp_port}
        self.info("ZeroMQ endpoints: %s",
                  ' '.join(sorted(self.zmq_endpoints.values())))

    def endpoint(self, mid, pid, hip):
        if self.mid == mid:
            if self.pid == pid:
                return self.zmq_endpoints["inproc"]
            else:
                return self.zmq_endpoints["ipc"]
        else:
            return self.zmq_endpoints["tcp"].replace("*", hip)

    def initialize(self):
        pass
