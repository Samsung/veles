"""
ZeroMQ Twisted factory which is controlling ZeroMQ context.
"""
from zmq import Context

from twisted.internet import reactor


class ZmqFactory(object):
    """
    I control individual ZeroMQ connections.

    Factory creates and destroys ZeroMQ context.

    :var reactor: reference to Twisted reactor used by all the connections
    :var ioThreads: number of IO threads ZeroMQ will be using for this context
    :vartype ioThreads: int
    :var lingerPeriod: number of milliseconds to block when closing socket
        (terminating context), when there are some messages pending to be sent
    :vartype lingerPeriod: int

    :var connections: set of instanciated :class:`ZmqConnection`
    :vartype connections: set
    :var context: ZeroMQ context
    """

    reactor = reactor
    ioThreads = 1
    lingerPeriod = 100

    def __init__(self):
        """
        Constructor.

        Create ZeroMQ context.
        """
        self.connections = set()
        self.context = Context(self.ioThreads)

    def __repr__(self):
        return "ZmqFactory()"

    def shutdown(self):
        """
        Shutdown factory.

        This is shutting down all created connections
        and terminating ZeroMQ context. Also cleans up
        Twisted reactor.
        """
        for connection in self.connections.copy():
            connection.shutdown()

        self.connections = None

        self.context.term()
        self.context = None

    def registerForShutdown(self):
        """
        Register factory to be automatically shut down
        on reactor shutdown.

        It is recommended that this method is called on any
        created factory.
        """
        reactor.addSystemEventTrigger('during', 'shutdown', self.shutdown)
