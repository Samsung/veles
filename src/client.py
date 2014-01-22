"""
Created on Jan 22, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from fysom import Fysom
import logging
from twisted.internet import reactor, threads
from twisted.internet.protocol import ReconnectingClientFactory
from twisted.protocols.basic import LineReceiver
import network_config


class VelesProtocol(LineReceiver):
    """A communication controller from client to server.

    Attributes:
        FSM_DESCRIPTION     The definition of the Finite State Machine of the
                            protocol.
    """
    FSM_DESCRIPTION = {
        'initial': 'INIT',
        'events': [
            {'name': 'connect', 'src': 'INIT', 'dst': 'WAIT'},
            {'name': 'receive_id', 'src': 'WAIT', 'dst': 'WORK'},
            {'name': 'send_id', 'src': 'WAIT', 'dst': 'WORK'},
            {'name': 'receive_error', 'src': '*', 'dst': 'ERROR'},
            {'name': 'disconnect_on_error', 'src': 'ERROR', 'dst': 'INIT'},
            {'name': 'reconnect', 'src': '*', 'dst': 'INIT'},
            {'name': 'request_job', 'src': 'WORK', 'dst': 'GETTING_JOB'},
            {'name': 'obtain_job', 'src': 'GETTING_JOB', 'dst': 'BUSY'},
            {'name': 'send_update', 'src': 'BUSY', 'dst': 'WORK'},
            {'name': 'refuse_job', 'src': 'GETTING_JOB', 'dst': 'WORK'},
            {'name': 'reconnect_busy', 'src': 'BUSY',
                                       'dst': 'RECONNECT_UPDATE'},
            {'name': 'wait_for_update_after_reconnection',
             'src': 'RECONNECT_UPDATE', 'dst': 'BUSY'},
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
        logging.info("state: %s, %s -> %s", e.event, e.src, e.dst)

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
        if not factory.state:
            factory.state = Fysom(VelesProtocol.FSM_DESCRIPTION)
        self.state = factory.state

    def connectionMade(self):
        if self.state.current == "INIT":
            if self.factory.id:
                self.send_id()
            else:
                self.sendLine("{{'power': %s}}" %
                              self.factory.host.get_computing_power())
            self.state.connect()
        elif self.state.current == "RECONNECT_UPDATE":
            if self.factory.id:
                self.send_id()
            else:
                logging.error("No id in RECONNECT_UPDATE state")
                self.disconnectOnError(False)
                return
            self.state.wait_for_update_after_reconnection()
            return
        logging.error("New connection contradicts with the current state.")

    def connectionLost(self, reason):
        logging.debug("Connection was lost.")
        if self.state.current == "ERROR":
            self.factory.must_reconnect = False
            self.state.disconnect_on_error()
            return
        if self.state.current == "BUSY":
            self.state.reconnect_busy()
            return
        self.state.reconnect()

    def lineReceived(self, line):
        logging.debug("%s:  %s", self.id, line)
        msg = eval(line)
        if not isinstance(msg, dict):
            logging.error("Could not parse the received line, dropping it.")
            return
        error = msg.get("error")
        if error:
            logging.error("Server returned error: '%s'.", error)
            self.disconnectOnError()
            return
        if self.state == "WAIT_ID":
            cid = msg.get("id")
            if not cid:
                logging.error("No id is present.")
                self.disconnectOnError()
                return
            self.factory.id = cid
            self.state.receive_id()
            return
        if self.state == "WORK":
            self.requestJob()
            self.state.request_job()
            return
        if self.state == "GETTING_JOB":
            job = msg.get("job")
            if not job:
                logging.error("No job is present.")
                self.disconnectOnError()
                return
            if job == "refuse":
                logging.info("Job was refused.")
                self.state.refuse_job()
                self.requestJob()
                return
            if job != "offer":
                logging.error("Unknown job value %s.", job)
                self.disconnectOnError(False)
                return
            self.setRawMode()
        logging.error("Invalid state %s.", self.state.current)

    def rawDataReceived(self, data):
        if self.state.current == 'GETTING_JOB':
            self.setLineMode()
            job = threads.deferToThreadPool(reactor,
                                            self.host.thread_pool(),
                                            self.factory.host.do_job,
                                            data)
            job.addCallback(self.onJobFinished)
            self.state.obtain_job()
            return
        logging.error("Invalid state %s.", self.state.current)

    def requestJob(self):
        self.sendLine("{{'cmd': 'job'}}")

    def disconnectOnError(self, reconnect=True):
        self.factory.must_reconnect = reconnect
        self.state.receive_error()
        self.transport.loseConnection()

    def onJobFinished(self, update):
        if self.state.current == "BUSY":
            self.sendLine("{{'cmd': 'update'}}")
            self.transport.write(update)
            self.state.send_update()
            return
        logging.error("Invalid state %s.", self.state.current)

    def send_id(self):
        self.sendLine("{{'id': %s}}" % self.factory.id)


class VelesProtocolFactory(ReconnectingClientFactory):
    def __init__(self, host):
        super(VelesProtocolFactory, self).__init__()
        self.host = host
        self.id = None
        self.must_reconnect = True
        self.state = None

    def startedConnecting(self, connector):
        logging.info('Connecting')

    def buildProtocol(self, addr):
        return VelesProtocol(addr, self)

    def clientConnectionLost(self, connector, reason):
        if self.must_reconnect:
            logging.warn("Disconnected, trying to reconnect...")
            connector.connect()
        else:
            logging.info("Disconnected.")

    def clientConnectionFailed(self, connector, reason):
        logging.warn('Connection failed. Reason: %s', reason)


class Client(network_config.NetworkConfigurable):
    """
    UDT/TCP client operating on a single socket.
    """

    def __init__(self, config_file, host):
        super(Client, self).__init__(config_file)
        self.factory = VelesProtocolFactory(host)
        reactor.connectTCP(self.address, self.port, self.factory)

    def run(self):
        reactor.run()

    def stop(self):
        reactor.stop()
