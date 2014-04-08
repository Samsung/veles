"""
Created on Apr 2, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from six.moves import cPickle as pickle, queue
from txzmq import ZmqConnection, ZmqEndpoint
import zmq


from veles.units import Unit


class ZmqPuller(ZmqConnection):
    socketType = zmq.PULL

    def __init__(self, owner, *endpoints):
        super(ZmqPuller, self).__init__(endpoints)
        self._owner = owner

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        self._owner = value

    def messageReceived(self, message):
        self.owner.data_received(pickle.loads(message))


class ZeroMQLoader(Unit):
    """
    Listens to incoming ZeroMQ sockets.
    """

    def __init__(self, workflow, *endpoints, **kwargs):
        super(ZeroMQLoader, self).__init__(workflow, **kwargs)
        self._endpoints = list(map(ZmqEndpoint, endpoints))
        self._queue = queue.Queue(kwargs.get("queue_size", 0))
        self.output = None

    @property
    def endpoints(self):
        return self._endpoints

    def initialize(self):
        super(ZeroMQLoader, self).initialize()
        self._zmq_socket = ZmqPuller(self, self.endpoints)

    def run(self):
        self.output = self._queue.get()

    def data_received(self, data):
        self._queue.put_nowait(data)
