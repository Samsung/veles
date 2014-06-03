"""
ZeroMQ connection.
"""
from collections import deque, namedtuple
import gzip
import sys
import six
from six.moves import cPickle as pickle
import os
import snappy
import lzma

from zmq import constants, error
from zmq import Socket

from zope.interface import implementer

from tempfile import mkstemp

from twisted.internet import reactor
from twisted.internet.interfaces import IFileDescriptor, IReadDescriptor
from twisted.python import log

from zmq import zmq_version_info
ZMQ3 = zmq_version_info()[0] >= 3

from .manager import ZmqContextManager


class ZmqEndpointType(object):
    """
    Endpoint could be "bound" or "connected".
    """
    bind = "bind"
    """
    Bind, listen for connection.
    """
    connect = "connect"
    """
    Connect to another endpoint.
    """


class ZmqEndpoint(namedtuple('ZmqEndpoint', ['type', 'address'])):
    """
    ZeroMQ endpoint used when connecting or listening for connections.

    Consists of two members: `type` and `address`.

    :var type: Could be either :attr:`ZmqEndpointType.bind` or
        :attr:`ZmqEndpointType.connect`.
    :var address: ZeroMQ address of endpoint, could be IP address,
        filename, see ZeroMQ docs for more details.
    :vartype address: str
    """


@implementer(IReadDescriptor, IFileDescriptor)
class ZmqConnection(object):
    """
    Connection through ZeroMQ, wraps up ZeroMQ socket.

    This class isn't supposed to be used directly, instead use one of the
    descendants like :class:`ZmqPushConnection`.

    :class:`ZmqConnection` implements glue between ZeroMQ and Twisted
    reactor: putting polling ZeroMQ file descriptor into reactor,
    processing events, reading data from socket.

    :var socketType: socket type, from ZeroMQ
    :var allowLoopbackMulticast: is loopback multicast allowed?
    :vartype allowLoopbackMulticast: bool
    :var multicastRate: maximum allowed multicast rate, kbps
    :vartype multicastRate: int
    :var highWaterMark: hard limit on the maximum number of outstanding
        messages 0MQ shall queue in memory for any single peer
    :vartype highWaterMark: int
    :var socket: ZeroMQ Socket
    :vartype socket: zmq.Socket
    :var endpoints: ZeroMQ addresses for connect/bind
    :vartype endpoints: list of :class:`ZmqEndpoint`
    :var fd: file descriptor of zmq mailbox
    :vartype fd: int
    :var queue: output message queue
    :vartype queue: deque
    """

    socketType = None
    allowLoopbackMulticast = False
    multicastRate = 100
    highWaterMark = 0

    # Only supported by zeromq3 and pyzmq>=2.2.0.1
    tcpKeepalive = 0
    tcpKeepaliveCount = 0
    tcpKeepaliveIdle = 0
    tcpKeepaliveInterval = 0

    PICKLE_START = b'vpb'
    PICKLE_END = b'vpe'
    CODECS = {None: b'\x00', "": b'\x00', "gzip": b'\x01', "snappy": b'\x02',
              "xz": b'\x03'}

    def __init__(self, endpoints, identity=None):
        """
        Constructor.

        :param factory: ZeroMQ Twisted factory
        :type factory: :class:`ZmqFactory`
        :param identity: socket identity (ZeroMQ), don't set unless you know
            how it works
        :type identity: str
        """
        self.factory = ZmqContextManager()
        self.endpoints = []
        self.identity = identity
        self.socket = Socket(self.factory.context, self.socketType)
        self.queue = deque()
        self.recv_parts = []
        self.read_scheduled = None
        self.shutted_down = False

        self.fd = self.socket.get(constants.FD)
        self.socket.set(constants.LINGER, self.factory.lingerPeriod)

        if not ZMQ3:
            self.socket.set(
                constants.MCAST_LOOP, int(self.allowLoopbackMulticast))

        self.socket.set(constants.RATE, self.multicastRate)

        if not ZMQ3:
            self.socket.set(constants.HWM, self.highWaterMark)
        else:
            self.socket.set(constants.SNDHWM, self.highWaterMark)
            self.socket.set(constants.RCVHWM, self.highWaterMark)

        if ZMQ3 and self.tcpKeepalive:
            self.socket.set(
                constants.TCP_KEEPALIVE, self.tcpKeepalive)
            self.socket.set(
                constants.TCP_KEEPALIVE_CNT, self.tcpKeepaliveCount)
            self.socket.set(
                constants.TCP_KEEPALIVE_IDLE, self.tcpKeepaliveIdle)
            self.socket.set(
                constants.TCP_KEEPALIVE_INTVL, self.tcpKeepaliveInterval)

        if self.identity is not None:
            self.socket.set(constants.IDENTITY, self.identity)

        self.endpoints = endpoints
        self.rnd_vals = self._connectOrBind(endpoints)

        self.factory.connections.add(self)

        self.factory.reactor.addReader(self)
        self.doRead()

    def shutdown(self):
        """
        Shutdown (close) connection and ZeroMQ socket.
        """
        if self.shutted_down:
            return
        self.shutted_down = True
        self.factory.reactor.removeReader(self)

        self.factory.connections.discard(self)

        self.socket.close()
        self.socket = None

        self.factory = None

        if self.read_scheduled is not None:
            self.read_scheduled.cancel()
            self.read_scheduled = None

    def __repr__(self):
        return "%s(%r, %r)" % (
            self.__class__.__name__, self.factory, self.endpoints)

    def fileno(self):
        """
        Implementation of :tm:`IFileDescriptor
        <internet.interfaces.IFileDescriptor>`.

        Returns ZeroMQ polling file descriptor.

        :return: The platform-specified representation of a file descriptor
            number.
        """
        return self.fd

    def connectionLost(self, reason):
        """
        Called when the connection was lost.

        Implementation of :tm:`IFileDescriptor
        <internet.interfaces.IFileDescriptor>`.

        This is called when the connection on a selectable object has been
        lost.  It will be called whether the connection was closed explicitly,
        an exception occurred in an event handler, or the other end of the
        connection closed it first.
        """
        if self.factory:
            self.factory.reactor.removeReader(self)

    def _readMultipart(self, unpickler):
        """
        Read multipart in non-blocking manner, returns with ready message
        or raising exception (in case of no more messages available).
        """
        while True:
            part = self.socket.recv(constants.NOBLOCK)
            if part.startswith(ZmqConnection.PICKLE_START):
                unpickler.active = True
                unpickler.codec = part[len(ZmqConnection.PICKLE_START)]
                continue
            if part == ZmqConnection.PICKLE_END:
                unpickler.active = False
                self.recv_parts.append(unpickler.object)
            elif not unpickler.active:
                self.recv_parts.append(part)
            else:
                unpickler.consume(part)
                continue
            if not self.socket.get(constants.RCVMORE):
                result, self.recv_parts = self.recv_parts, []

                return result

    def doRead(self):
        """
        Some data is available for reading on ZeroMQ descriptor.

        ZeroMQ is signalling that we should process some events,
        we're starting to receive incoming messages.

        Implementation of :tm:`IReadDescriptor
        <internet.interfaces.IReadDescriptor>`.
        """
        if self.shutted_down:
            return
        if self.read_scheduled is not None:
            if not self.read_scheduled.called:
                self.read_scheduled.cancel()
            self.read_scheduled = None

        class Unpickler(object):
            def __init__(self):
                self._active = False

            @property
            def active(self):
                return self._active

            @active.setter
            def active(self, value):
                self._active = value
                if not value:
                    size = sum([len(d) for d in self._data])
                    buffer = bytearray(size)
                    pos = 0
                    for d in self._data:
                        buffer[pos:pos + len(d)] = d
                    self._object = pickle.loads(buffer if six.PY3
                                                else str(buffer))
                self._data = []

            @property
            def codec(self):
                return self._codec

            @codec.setter
            def codec(self, value):
                self._codec = value if six.PY3 else ord(value)

            @property
            def object(self):
                return self._object

            def consume(self, data):
                codec = self._codec
                if codec == 2:
                    chunk = snappy.decompress(data)
                elif codec == 0:
                    chunk = data
                elif codec == 1:
                    chunk = gzip.decompress(data)
                elif codec == 3:
                    chunk = lzma.decompress(data)
                else:
                    raise RuntimeError("Unknown compression type")
                self._data.append(chunk)

        unpickler = Unpickler()
        while True:
            if self.factory is None:  # disconnected
                return

            events = self.socket.get(constants.EVENTS)

            if (events & constants.POLLIN) != constants.POLLIN:
                return

            try:
                message = self._readMultipart(unpickler)
            except error.ZMQError as e:
                if e.errno == constants.EAGAIN:
                    continue

                raise e

            log.callWithLogger(self, self.messageReceived, message)

    def logPrefix(self):
        """
        Implementation of :tm:`ILoggingContext
        <internet.interfaces.ILoggingContext>`.

        :return: Prefix used during log formatting to indicate context.
        :rtype: str
        """
        return 'ZMQ'

    def send(self, *message, pickles_compression="snappy"):
        """
        Send message via ZeroMQ socket.

        Sending is performed directly to ZeroMQ without queueing. If HWM is
        reached on ZeroMQ side, sending operation is aborted with exception
        from ZeroMQ (EAGAIN).

        After writing read is scheduled as ZeroMQ may not signal incoming
        messages after we touched socket with write request.

        :param message: message data, a series of objects; if an object is
        an instance of bytes, it will be sent as-is, otherwise, it will be
        pickled and optionally compressed. Object must not be a string.
        :param pickles_compression: the compression to apply to pickled
        objects. Supported values are None or "", "gzip", "snappy" and "xz".
        :type pickles_compression: str
        """
        if self.shutted_down:
            return

        def send_part(msg, last):
            flag = constants.SNDMORE if not last else 0
            if isinstance(msg, bytes):
                self.socket.send(msg, constants.NOBLOCK | flag)
            elif isinstance(msg, str):
                raise ValueError("All strings must be encoded into bytes")
            else:
                self._send_pickled(msg, last, pickles_compression)

        for m in message[:-1]:
            send_part(m, False)
        send_part(message[-1], True)

        if self.read_scheduled is None:
            self.read_scheduled = reactor.callLater(0, self.doRead)

    def _send_pickled(self, message, last, compression):
        if self.shutted_down:
            return

        class Pickler(object):
            def __init__(self, socket, codec):
                self._socket = socket
                self._codec = codec if six.PY3 else ord(codec)

            def write(self, data):
                codec = self._codec
                if codec == 2:
                    chunk = snappy.compress(data)
                elif codec == 0:
                    chunk = data
                elif codec == 1:
                    chunk = gzip.compress(data)
                elif codec == 3:
                    chunk = lzma.compress(data)
                else:
                    raise RuntimeError("Unknown compression type")
                self._socket.send(chunk, constants.NOBLOCK | constants.SNDMORE)

        codec = ZmqConnection.CODECS.get(compression)
        if codec is None:
            raise ValueError("Unknown compression type: %s" % compression)
        pickler = Pickler(self.socket, codec[0])
        self.socket.send(ZmqConnection.PICKLE_START + codec,
                         constants.NOBLOCK | constants.SNDMORE)
        pickle.dump(message, pickler, protocol=(4 if sys.version_info > (3, 4)
                                                else 3 if six.PY3 else 2))
        self.socket.send(
            ZmqConnection.PICKLE_END,
            constants.NOBLOCK | (constants.SNDMORE if not last else 0))

    def messageReceived(self, message):
        """
        Called when complete message is received.

        Not implemented in :class:`ZmqConnection`, should be overridden to
        handle incoming messages.

        :param message: message data
        """
        raise NotImplementedError(self)

    def _connectOrBind(self, endpoints):
        """
        Connect and/or bind socket to endpoints.
        """
        rnd_vals = []
        for endpoint in endpoints:
            if endpoint.type == ZmqEndpointType.connect:
                self.socket.connect(endpoint.address)
            elif endpoint.type == ZmqEndpointType.bind:
                if endpoint.address.startswith("rndtcp://") or \
                   endpoint.address.startswith("rndepgm://"):
                    endpos = endpoint.address.find("://") + 3
                    proto = endpoint.address[3:endpos]
                    addr, min_port, max_port, max_tries = \
                        endpoint.address[endpos:].split(':')
                    rnd_vals.append(self.socket.bind_to_random_port(
                        proto + addr, int(min_port), int(max_port),
                        int(max_tries)))
                elif endpoint.address.startswith("rndipc://"):
                    prefix, suffix = endpoint.address[9:].split(':')
                    ipc_fd, ipc_fn = mkstemp(suffix, prefix)
                    self.socket.bind("ipc://" + ipc_fn)
                    rnd_vals.append(ipc_fn)
                    os.close(ipc_fd)
                else:
                    self.socket.bind(endpoint.address)
            else:
                assert False, "Unknown endpoint type %r" % endpoint
        return rnd_vals
