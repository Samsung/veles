"""
Created on Jan 14, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import fysom
import getpass
import json
import socket
import time
import threading
from tornado.httpclient import AsyncHTTPClient
from tornado.ioloop import IOLoop
from twisted.internet import reactor, threads, task
from twisted.internet.protocol import Factory
from twisted.web.html import escape
from txzmq import ZmqConnection, ZmqEndpoint, ZmqEndpointType, ZmqFactory
import uuid
import zmq

import config
from daemon import daemonize
from logger import Logger
from network_common import NetworkAgent, StringLineReceiver
from graphics import Graphics


class ZmqPusher(ZmqConnection):
    socketType = zmq.constants.PUSH


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

    def onJobObtained(self, e):
        self.nodes[self.id]["state"] = "Working"

    def setWaiting(self, e):
        self.nodes[self.id]["state"] = "Waiting"

    def onDropped(self, e):
        if self.id in self.nodes:
            self.nodes[self.id]["state"] = "Offline"

    FSM_DESCRIPTION = {
        'initial': 'INIT',
        'events': [
            {'name': 'connect', 'src': 'INIT', 'dst': 'WAIT'},
            {'name': 'receive_description', 'src': 'WAIT', 'dst': 'WORK'},
            {'name': 'receive_id', 'src': 'WAIT', 'dst': 'WORK'},
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
            'onobtain_job': onJobObtained,
            'onreceive_update': setWaiting,
            'onWORK': setWaiting,
            'ondrop': onDropped
        }
    }

    def __init__(self, addr, nodes, host):
        """
        Initializes the protocol.

        Parameters:
            addr    The address of the client (reported by Twisted).
            nodes   The clients which are known (dictionary, the key is ID).
            host    An instance of MasterWorkflow which uses the server.
        """
        super(VelesProtocol, self).__init__()
        self.addr = addr
        self.nodes = nodes
        self.host = host
        self.id = None
        self.state = fysom.Fysom(VelesProtocol.FSM_DESCRIPTION, self)

    def disappear(self):
        del(self.nodes[self.id])

    def connectionMade(self):
        self.state.connect()

    def connectionLost(self, reason):
        self.state.drop()
        if self.host.workflow.is_finished():
            self.disappear()
            if len(self.nodes) == 0:
                self.host.stop()
        else:
            upd = threads.deferToThreadPool(reactor,
                                            self.host.workflow.thread_pool(),
                                            self.host.workflow.drop_slave,
                                            self.nodes[self.id])
            upd.addCallback(self.disappear)

    def lineReceived(self, line):
        self.host.debug("%s lineReceived:  %s", self.id, line)
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            self.host.error("%s Could not parse the received line, dropping "
                            "it.", self.id)
            return
        if self.state.current == "WAIT":
            mysha = self.host.workflow.checksum()
            your_sha = msg.get("checksum")
            if not your_sha:
                self.host.error("Did not receive the workflow checksum.")
                self.sendError("Workflow checksum is missing.")
                return
            if mysha != your_sha:
                self.host.error("Workflow checksum mismatch.")
                self.sendError("Workflow checksum mismatched.")
                return
            cid = msg.get("id")
            if not cid:
                power = msg.get("power")
                if not power:
                    self.host.error("Newly connected client did not send "
                                    "it's computing power value, sending back "
                                    "the error message.")
                    self.sendError("I need your computing power.")
                    return
                self.id = str(uuid.uuid4())
                self.nodes[self.id] = {'power': power}
                reactor.callLater(0, self.resolveAddr, self.addr)
                self.sendLine({'id': self.id})
                self.state.receive_description()
                return
            self.id = cid
            if not self.nodes.get(cid):
                self.host.warning("Did not recognize the received ID %s.", cid)
                power = msg.get("power")
                if not power:
                    self.host.error("%s Unable to add a client without it's "
                                    "power.", self.id)
                    self.sendError("I need your computing power.")
                    return
                self.nodes[self.id] = {'power': power}
                reactor.callLater(0, self.resolveAddr, self.addr)
            self.state.receive_id()
        if self.state.current == "WORK":
            cmd = msg.get("cmd")
            if not cmd:
                self.host.error("%s Client sent something which is not "
                                "a command: %s. Sending back the error "
                                "message.", self.id, line)
                self.sendError("No command found.")
                return
            if cmd == "update":
                self.host.debug("%s Received UPDATE command. "
                                "Expecting to receive a pickle.", self.id)
                self.size = msg.get("size")
                if self.size == None:
                    self.disconnect("Update size was not specified.")
                    return
                self.update = bytearray(self.size)
                self.update_pos = 0
                self.setRawMode()
                self.state.receive_update()
            elif cmd == "job":
                self.host.debug("%s Received JOB command. " +
                                "Requesting a new job from the host.", self.id)
                job = threads.deferToThreadPool(
                    reactor,
                    self.host.workflow.thread_pool(),
                    self.host.workflow.request_job)
                job.addCallback(self.jobRequestFinished)
                self.state.request_job()
            else:
                self.host.error("%s Unsupported %s command. Sending back the "
                                "error message.", self.id, cmd)
                self.sendError("Unsupported command.")
            return
        self.host.error("%s Invalid state %s.", self.id, self.state.current)
        self.sendError("You sent me something which is not allowed in my " +
                       "current state %s." % self.state.current)

    def rawDataReceived(self, data):
        if self.state.current == 'APPLYING_UPDATE':
            self.update[self.update_pos:self.update_pos + len(data)] = data
            self.update_pos += len(data)
            if self.update_pos == self.size:
                self.setLineMode()
                upd = threads.deferToThreadPool(
                    reactor,
                    self.host.workflow.thread_pool(),
                    self.host.workflow.apply_update,
                    self.update, self.nodes[self.id])
                upd.addCallback(self.updateApplied)
            if len(self.update) > self.size:
                self.disconnect("Received update size %d exceeded the expected"
                                " length (%d)", len(self.update), self.size)
        else:
            self.host.error("%s Cannot receive raw data in %s state.",
                            self.id, self.state.current)

    def updateApplied(self, result):
        if self.state.current == 'APPLYING_UPDATE':
            if result:
                self.sendLine({'update': 'ok'})
            else:
                self.sendLine({'update': 'deny'})
            self.state.apply_update()
        else:
            self.host.error("%s Wrong state for update.", self.id)

    def jobRequestFinished(self, data=None):
        if data != None:
            self.sendLine({'job': 'offer', 'size': len(data)})
            self.host.debug("%s Job size: %d Kb", self.id, len(data) / 1000)
            self.transport.write(data)
            self.state.obtain_job()
        else:
            self.sendLine({'job': 'refuse'})
            self.state.refuse_job()

    def resolveAddr(self, addr):
        host, _, _ = socket.gethostbyaddr(addr.host)
        if host == "localhost":
            host = socket.gethostname()
        self.host.debug("%s Address %s was resolved to %s", self.id,
                        addr.host, host)
        self.nodes[self.id]['host'] = host

    def sendLine(self, line):
        super(VelesProtocol, self).sendLine(json.dumps(line))

    def sendError(self, err):
        """
        Sends the line with the specified error message.

        Parameters:
            err:    The error message.
        """
        self.host.error(err)
        self.sendLine({'error': err})


class VelesProtocolFactory(Factory):
    def __init__(self, host):
        super(VelesProtocolFactory, self).__init__()
        self.nodes = {}
        self.host = host

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self.nodes, self.host)


class Server(NetworkAgent, Logger):
    """
    UDT/TCP server operating on a single socket
    """

    def __init__(self, configuration, workflow):
        super(Server, self).__init__(configuration)
        self.id = str(uuid.uuid4())
        self.workflow = workflow
        self.workflow_graph = self.workflow.generate_graph(write_on_disk=False)
        self.factory = VelesProtocolFactory(self)
        reactor.listenTCP(self.port, self.factory, interface=self.address)
        self.notify_task = task.LoopingCall(self.notify_status)
        self.notify_agent = AsyncHTTPClient()
        self.tornado_ioloop_thread = threading.Thread(
            target=self.tornado_ioloop)
        self.zmq_factory = ZmqFactory()
        self.zmq_connection = ZmqPusher(self.zmq_factory)
        self.zmq_ipc_fn, self.zmq_tcp_port = self.zmq_connection.addEndpoints(
            ZmqEndpoint(ZmqEndpointType.bind, "inproc://veles"),
            ZmqEndpoint(ZmqEndpointType.bind, "rndipc://veles-ipc-:"),
            ZmqEndpoint(ZmqEndpointType.bind, "rndtcp://*:1024:65535:1"),
        )
        self.info("ZeroMQ endpoints: inproc://veles, ipc://%s, tcp://*:%d",
                  self.zmq_ipc_fn, self.zmq_tcp_port)

    @daemonize
    def run(self):
        self.start_time = time.time()
        self.tornado_ioloop_thread.start()
        self.notify_task.start(config.web_status_notification_interval,
                               now=False)
        try:
            reactor.run()
        except:
            self.exception("Failed to run the reactor")

    def stop(self):
        try:
            self.notify_task.stop()
            if reactor.running:
                reactor.stop()

            IOLoop.instance().stop()
            self.tornado_ioloop_thread.join()
        except:
            self.exception("Failed to stop the reactor")

    def handle_notify_request(self, response):
        if response.error:
            self.warning("Failed to upload the status update to %s:%s",
                         config.web_status_host, config.web_status_port)
        else:
            self.debug("Successfully updated the status")

    def tornado_ioloop(self):
        IOLoop.instance().start()

    def notify_status(self):
        if not IOLoop.instance().running():
            return
        mins, secs = divmod(time.time() - self.start_time, 60)
        hours, mins = divmod(mins, 60)
        ret = {'id': self.id,
               'name': self.workflow.name(),
               'master': socket.gethostname(),
               'time': "%02d:%02d:%02d" % (hours, mins, secs),
               'user': getpass.getuser(),
               'graph': self.workflow_graph,
               'slaves': self.factory.nodes,
               'plots': "http://" + socket.gethostname() + ":" +
                        str(Graphics.matplotlib_webagg_listened_port),
               'description': "<br />".join(escape(
                                  self.workflow.__doc__).split("\n"))}
        timeout = config.web_status_notification_interval / 2
        self.notify_agent.fetch("http://%s:%d/%s" % (config.web_status_host,
                                                     config.web_status_port,
                                                     config.web_status_update),
                                self.handle_notify_request,
                                method='POST', headers=None,
                                connect_timeout=timeout,
                                request_timeout=timeout,
                                body=json.dumps(ret))
