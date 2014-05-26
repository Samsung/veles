"""
Created on Apr 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from six.moves import cPickle as pickle, queue
from veles.external.txzmq import ZmqConnection, ZmqEndpoint
import zmq
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
from veles.units import Unit, IUnit


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


@implementer(IUnit)
class ZeroMQLoader(Unit, TriviallyDistributable):
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

    def initialize(self, **kwargs):
        super(ZeroMQLoader, self).initialize(**kwargs)
        self._zmq_socket = ZmqPuller(self, self.endpoints)

    def run(self):
        self.output = self._queue.get()

    def data_received(self, data):
        self._queue.put_nowait(data)
