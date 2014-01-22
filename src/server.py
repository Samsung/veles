"""
Created on Jan 14, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from fysom import Fysom
import logging
from twisted.internet import reactor, threads
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver
import uuid
import network_config


class VelesProtocol(LineReceiver):
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
        'callbacks': [
            {'onchangestate': VelesProtocol.onFSMStateChanged}
        ]
    }

    @staticmethod
    def onFSMStateChanged(e):
        """
        Logs the current state transition.
        """
        logging.info("%s state: %s, %s -> %s",
                     e.fsm.protocol.id, e.event, e.src, e.dst)

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
        msg = eval(line)
        if not isinstance(msg, dict):
            logging.error("Could not parse the received line, dropping it.")
            return
        if self.state == "WAIT":
            cid = msg.get("id")
            if not cid:
                power = msg.get("power")
                if not power:
                    logging.error("Newly connected client did not send it's " +
                                  "computing power value, sending back the" +
                                  " error message.")
                    self.sendError("I need your computing power.")
                    return
                self.id = uuid.uuid4()
                self.nodes[self.id] = {{'power': power}}
                self.sendLine("{{'id': '%s'}}" % self.id)
                self.state.receive_description()
                return
            if not self.nodes.get(cid):
                logging.error(("Did not recognize the received ID %s." +
                              "Sending back the error message.") % cid)
                self.sendError("Your ID was not found.")
            self.id = cid
            self.state.receive_id()
        if self.state == "WORK":
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
                self.state.receive_update()
                self.setRawMode()
            elif cmd == "job":
                logging.debug("Received JOB command. " +
                              "Requesting a new job from the host.")
                job = threads.deferToThreadPool(reactor,
                                                self.host.thread_pool(),
                                                self.host.request_job)
                job.addCallback(self.onJobRequestFinished)
                self.state.request_job()
            else:
                logging.error(("Unsupported %s command. Sending back the " +
                              "error message.") % cmd)
                self.sendError("Unsupported command.")
        logging.error("Invalid state %s.", self.state.current)
        self.sendError("You sent me something which is not allowed in my " +
                       "current state %s." % self.state.current)

    def rawDataReceived(self, data):
        if self.state.current == 'APPLYING_UPDATE':
            self.setLineMode()
            self.host.apply_update(data)
            self.state.apply_update()
        else:
            logging.error("Cannot receive raw data in %s state." %
                          self.state.current)

    def onJobRequestFinished(self, data=None):
        if data:
            self.sendLine("{{'job': 'offer'}}")
            self.transport.write(data)
            self.state.obtain_job()
        else:
            self.sendLine("{{'job': 'refuse'}}")
            self.refuse_job()

    def sendError(self, err):
        """
        Sends the line with the specified error message.

        Parameters:
            err:    The error message.
        """
        msg = "{{'error': '%s'}}" % err
        logging.debug("Sending " + msg)
        self.sendLine(msg)


class VelesProtocolFactory(Factory):
    def __init__(self, host):
        super(VelesProtocolFactory, self).__init__()
        self.nodes = []
        self.host = host

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self.nodes, self.host)


class Server(network_config.NetworkConfigurable):
    """
    UDT/TCP server operating on a single socket
    """

    def __init__(self, config_file, host):
        super(Server, self).__init__(config_file)
        self.factory = VelesProtocolFactory(host)
        reactor.listenTCP(self.port, self.factory)

    def run(self):
        reactor.run()

    def stop(self):
        reactor.stop()
