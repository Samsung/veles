"""
Created on Jan 14, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import json
from fysom import Fysom
import logging
from tornado.httpclient import AsyncHTTPClient
from twisted.internet import reactor, threads, task
from twisted.internet.protocol import Factory
import uuid
import socket

import config
import daemon
import network_common


def onFSMStateChanged(e):
        """
        Logs the current state transition.
        """
        owner_id = "None"
        if hasattr(e.fsm, "protocol"):
            owner_id = e.fsm.protocol.id
        logging.info("master %s state: %s, %s -> %s",
                     owner_id, e.event, e.src, e.dst)


class VelesProtocol(network_common.StringLineReceiver):
    """A communication controller from server to client.

    Attributes:
        FSM_DESCRIPTION     The definition of the Finite State Machine of the
                            protocol.
    """

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
        self.state = Fysom(VelesProtocol.FSM_DESCRIPTION)
        setattr(self.state, "protocol", self)

    def connectionMade(self):
        self.state.connect()

    def connectionLost(self, reason):
        self.state.drop()

    def lineReceived(self, line):
        logging.debug("lineReceived %s:  %s", self.id, line)
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            logging.error("Could not parse the received line, dropping it.")
            return
        if self.state.current == "WAIT":
            cid = msg.get("id")
            if not cid:
                power = msg.get("power")
                if not power:
                    logging.error("Newly connected client did not send it's " +
                                  "computing power value, sending back the" +
                                  " error message.")
                    self.sendError("I need your computing power.")
                    return
                self.id = str(uuid.uuid4())
                self.nodes[self.id] = {'power': power}
                reactor.callLater(0, self.resolveAddr, self.addr)
                self.sendLine({'id': self.id})
                self.state.receive_description()
                return
            if not self.nodes.get(cid):
                logging.error(("Did not recognize the received ID %s." +
                              "Sending back the error message.") % cid)
                self.sendError("Your ID was not found.")
            self.id = cid
            self.state.receive_id()
        if self.state.current == "WORK":
            cmd = msg.get("cmd")
            if not cmd:
                logging.error(("Client %s sent something which is not " +
                              "a command: %s. Sending back the error " +
                              "message.") % (self.id, line))
                self.sendError("No command found.")
                return
            if cmd == "update":
                logging.debug("Received UPDATE command. " +
                              "Expecting to receive a pickle.")
                self.setRawMode()
                self.state.receive_update()
            elif cmd == "job":
                logging.debug("Received JOB command. " +
                              "Requesting a new job from the host.")
                job = threads.deferToThreadPool(reactor,
                                                self.host.thread_pool(),
                                                self.host.request_job)
                job.addCallback(self.jobRequestFinished)
                self.state.request_job()
            else:
                logging.error(("Unsupported %s command. Sending back the " +
                              "error message.") % cmd)
                self.sendError("Unsupported command.")
            return
        logging.error("Invalid state %s.", self.state.current)
        self.sendError("You sent me something which is not allowed in my " +
                       "current state %s." % self.state.current)

    def rawDataReceived(self, data):
        if self.state.current == 'APPLYING_UPDATE':
            self.setLineMode()
            upd = threads.deferToThreadPool(reactor,
                                            self.host.thread_pool(),
                                            self.host.apply_update,
                                            data)
            upd.addCallback(self.updateApplied)

        else:
            logging.error("Cannot receive raw data in %s state." %
                          self.state.current)

    def updateApplied(self, result):
        if self.state.current == 'APPLYING_UPDATE':
            if result:
                self.sendLine({'update': 'ok'})
            else:
                self.sendLine({'update': 'deny'})
            self.state.apply_update()
        else:
            logging.error("Wrong state.")

    def jobRequestFinished(self, data=None):
        if data:
            self.sendLine({'job': 'offer'})
            self.transport.write(data)
            self.state.obtain_job()
        else:
            self.sendLine({'job': 'refuse'})
            self.refuse_job()

    def resolveAddr(self, addr):
        host, _, _ = socket.gethostbyaddr(addr.host)
        logging.debug("Address %s was resolved to %s", addr.host, host)
        self.nodes[self.id]['host'] = host

    def sendLine(self, line):
        super(VelesProtocol, self).sendLine(json.dumps(line))

    def sendError(self, err):
        """
        Sends the line with the specified error message.

        Parameters:
            err:    The error message.
        """
        logging.error(err)
        self.sendLine({'error': err})


class VelesProtocolFactory(Factory):
    def __init__(self, host):
        super(VelesProtocolFactory, self).__init__()
        self.nodes = {}
        self.host = host

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self.nodes, self.host)


class Server(network_common.NetworkConfigurable):
    """
    UDT/TCP server operating on a single socket
    """

    def __init__(self, configuration, workflow):
        super(Server, self).__init__(configuration)
        self.workflow = workflow
        self.workflow_graph = self.workflow.generate_graph(write_on_disk=False)
        self.factory = VelesProtocolFactory(workflow)
        reactor.listenTCP(self.port, self.factory, interface=self.address)
        self.notify_task = task.LoopingCall(self.notify_status)
        self.notify_agent = AsyncHTTPClient()

    def run(self, daemon=False):
        with daemon.DaemonContext() if daemon else None:
            self.notify_task.start(config.web_status_notification_interval,
                                   now=False)
            reactor.run()

    def stop(self):
        self.notify_task.stop()
        reactor.stop()

    def handle_notify_request(self, response):
        if response.error:
            logging.warn("Failed to upload the status update to %s:%s",
                         config.web_status_host, config.web_status_port)

    def notify_status(self):
        ret = {'nodes': {}, 'workflow': {}}
        ret['workflow']['graph'] = self.workflow_graph
        for nid, node in self.factory.nodes.items():
            ret['nodes'][nid] = {'host': node["host"]}
        self.notify_agent.fetch("http://%s/%s:%d" % (config.web_status_host,
                                                     config.web_status_update,
                                                     config.web_status_port),
                                self.handle_notify_request,
                                method='POST', headers=None,
                                connect_timeout=0.2,
                                body=json.dumps(ret))
