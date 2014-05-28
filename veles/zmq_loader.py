"""
Created on Apr 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from six.moves import cPickle as pickle, queue
from veles.external.txzmq import ZmqConnection, ZmqEndpoint
import zmq
from zope.interface import implementer

from veles.distributable import IDistributable
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
        self.owner.receive_data(pickle.loads(message[0]))


@implementer(IUnit, IDistributable)
class ZeroMQLoader(Unit):
    """
    Listens to incoming ZeroMQ sockets.
    """

    def __init__(self, workflow, **kwargs):
        super(ZeroMQLoader, self).__init__(workflow, **kwargs)
        self._queue = queue.Queue(kwargs.get("queue_size", 0))
        self.output = 0
        self._endpoints = {}
        self.negotiates_on_connect = True

    @property
    def endpoints(self):
        return self._endpoints

    def initialize(self, **kwargs):
        if not self.is_slave:
            return
        self.endpoints.update({
            "inproc":
            ZmqEndpoint("bind", "inproc://veles-zmqloader-%s" % self.name),
            "ipc":
            ZmqEndpoint("bind", "rndipc://veles-ipc-zmqloader-:"),
            "tcp":
            ZmqEndpoint("bind", "rndtcp://*:1024:65535:1")})
        self._zmq_socket = ZmqPuller(self, *sorted(self.endpoints.values()))

        zmq_ipc_fn, zmq_tcp_port = self._zmq_socket.rnd_vals
        self.endpoints.update({
            "inproc":
            ZmqEndpoint("connect", self.endpoints['inproc'].address),
            "ipc":
            ZmqEndpoint("connect", "ipc://%s" % zmq_ipc_fn),
            "tcp":
            ZmqEndpoint("connect", "tcp://*:%d" % zmq_tcp_port)})

    def run(self):
        self.output = self._queue.get()

    def stop(self):
        self.receive_data(None)

    def receive_data(self, data):
        self._queue.put_nowait(data)

    def apply_data_from_slave(self, data, slave):
        self._endpoints[slave.id] = (slave, data["ZmqLoaderEndpoints"])

    def apply_data_from_master(self, data):
        pass

    def generate_data_for_master(self):
        return {"ZmqLoaderEndpoints": self._endpoints}

    def generate_data_for_slave(self):
        return None

    def drop_slave(self, slave):
        del self._endpoints[slave.id]
