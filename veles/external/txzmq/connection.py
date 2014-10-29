"""
ZeroMQ connection.
"""
from collections import deque, namedtuple
import gzip
import zlib
import sys
import six
from six.moves import cPickle as pickle
import os
import snappy
import time

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
from .sharedio import SharedIO

from veles.compat import lzma, from_none
from veles.logger import Logger


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
class ZmqConnection(Logger):
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

    class IOOverflow(Exception):
        pass

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

    def __init__(self, endpoints, identity=None, **kwargs):
        """
        Constructor.

        :param factory: ZeroMQ Twisted factory
        :type factory: :class:`ZmqFactory`
        :param identity: socket identity (ZeroMQ), don't set unless you know
            how it works
        :type identity: str
        """
        super(ZmqConnection, self).__init__()
        self.factory = ZmqContextManager()
        self.endpoints = []
        self.identity = identity
        self.socket = Socket(self.factory.context, self.socketType)
        self.queue = deque()
        self.recv_parts = []
        self.read_scheduled = None
        self.shutted_down = False
        self.pickles_compression = "snappy"
        self._last_read_time = 0.0

        self.fd = self.socket.get(constants.FD)
        self.socket.set(constants.LINGER, self.factory.lingerPeriod)
        self.socket.set(constants.RATE, self.multicastRate)
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

    @property
    def pickles_compression(self):
        return self._pickles_compression

    @pickles_compression.setter
    def pickles_compression(self, value):
        if value not in (None, "", "gzip", "snappy", "xz"):
            raise ValueError()
        self._pickles_compression = value

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
                self.messageHeaderReceived(self.recv_parts)
                unpickler.active = True
                unpickler.codec = part[len(ZmqConnection.PICKLE_START)]
                continue
            if part == ZmqConnection.PICKLE_END:
                unpickler.active = False
                obj = unpickler.object
                if isinstance(obj, SharedIO):
                    obj = pickle.load(obj)
                self.recv_parts.append(obj)
            elif not unpickler.active:
                self.recv_parts.append(part)
            else:
                unpickler.consume(part)
                continue
            if not self.socket.get(constants.RCVMORE):
                result, self.recv_parts = self.recv_parts, []

                return result

    class Unpickler(object):
        def __init__(self):
            self._data = []
            self._active = False
            self._decompressor = None

        @property
        def active(self):
            return self._active

        @active.setter
        def active(self, value):
            self._active = value
            if not value:
                buffer = self.merge_chunks()
                self._object = pickle.loads(buffer if six.PY3 else str(buffer))
            self._data = []

        @property
        def codec(self):
            return self._codec

        @codec.setter
        def codec(self, value):
            self._codec = value if six.PY3 else ord(value)
            if self.codec == 0:
                pass
            elif self.codec == 1:
                self._decompressor = \
                    zlib.decompressobj(16 + zlib.MAX_WBITS)
            elif self.codec == 2:
                self._decompressor = snappy.StreamDecompressor()
            elif self.codec == 3:
                self._decompressor = lzma.LZMADecompressor()
            else:
                raise ValueError("Unknown compression type")

        @property
        def object(self):
            return self._object

        def merge_chunks(self):
            if self.codec > 0 and not isinstance(self._decompressor,
                                                 lzma.LZMADecompressor):
                self._data.append(self._decompressor.flush())
            size = sum([len(d) for d in self._data])
            buffer = bytearray(size)
            pos = 0
            for d in self._data:
                ld = len(d)
                buffer[pos:pos + ld] = d
                pos += ld
            return buffer

        def consume(self, data):
            if self.codec > 0:
                data = self._decompressor.decompress(data)
            self._data.append(data)

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

        unpickler = ZmqConnection.Unpickler()
        while True:
            if self.factory is None:  # disconnected
                return

            events = self.socket.get(constants.EVENTS)

            if (events & constants.POLLIN) != constants.POLLIN:
                return

            timestamp = time.time()
            try:
                message = self._readMultipart(unpickler)
            except error.ZMQError as e:
                if e.errno == constants.EAGAIN:
                    continue

                raise e
            finally:
                self._last_read_time = time.time() - timestamp
            log.callWithLogger(self, self.messageReceived, message)

    @property
    def last_read_time(self):
        return self._last_read_time

    def logPrefix(self):
        """
        Implementation of :tm:`ILoggingContext
        <internet.interfaces.ILoggingContext>`.

        :return: Prefix used during log formatting to indicate context.
        :rtype: str
        """
        return 'ZMQ'

    def send(self, *message, **kwargs):
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
        :param io: a SharedIO object where to put pickles into instead of the
        socket. Can be None.
        """
        if self.shutted_down:
            return
        pickles_compression = kwargs.get("pickles_compression", "snappy")
        pickles_size = 0
        io = kwargs.get("io")
        io_overflow = False

        def send_part(msg, last):
            flag = constants.SNDMORE if not last else 0
            if isinstance(msg, bytes):
                self.socket.send(msg, constants.NOBLOCK | flag)
                return 0
            if isinstance(msg, str):
                raise ValueError("All strings must be encoded into bytes")
            return self._send_pickled(msg, last, pickles_compression,
                                      io if not io_overflow else None)

        for i, m in enumerate(message):
            try:
                pickles_size += send_part(m, i == len(message) - 1)
            except ZmqConnection.IOOverflow:
                io_overflow = True

        if self.read_scheduled is None:
            self.read_scheduled = reactor.callLater(0, self.doRead)
        if io_overflow:
            raise ZmqConnection.IOOverflow()
        return pickles_size

    class SocketFile(object):
        def __init__(self, socket):
            self._socket = socket
            self._size = 0

        @property
        def size(self):
            return self._size

        @property
        def mode(self):
            return "wb"

        def write(self, data):
            self._size += len(data)
            self._socket.send(data,
                              constants.NOBLOCK | constants.SNDMORE)

        def flush(self):
            pass

    class CompressedFile(object):
        def __init__(self, fileobj, compressor):
            self._file = fileobj
            self._compressor = compressor

        @property
        def mode(self):
            return "wb"

        def write(self, data):
            self._file.write(self._compressor.compress(data))

        def flush(self):
            last = self._compressor.flush()
            if last is not None:
                self._file.write(last)
            self._file.flush()

    class Pickler(object):
        def __init__(self, socket, codec):
            self._codec = codec if six.PY3 else ord(codec)
            self._socketobj = ZmqConnection.SocketFile(socket)
            if self.codec == 0:
                self._compressor = self._socketobj
            elif self.codec == 1:
                self._compressor = gzip.GzipFile(fileobj=self._socketobj)
            elif self.codec == 2:
                self._compressor = ZmqConnection.CompressedFile(
                    self._socketobj, snappy.StreamCompressor())
            elif self.codec == 3:
                self._compressor = ZmqConnection.CompressedFile(
                    self._socketobj, lzma.LZMACompressor(lzma.FORMAT_XZ))
            else:
                raise ValueError("Unknown compression type")

        @property
        def size(self):
            return self._socketobj.size

        @property
        def codec(self):
            return self._codec

        @property
        def mode(self):
            return "wb"

        def write(self, data):
            self._compressor.write(data)

        def flush(self):
            self._compressor.flush()

    def _send_pickled(self, message, last, compression, io):
        if self.shutted_down:
            return

        codec = ZmqConnection.CODECS.get(compression)
        if codec is None:
            raise ValueError("Unknown compression type: %s" % compression)

        def send_pickle_beg_marker(codec):
            self.socket.send(ZmqConnection.PICKLE_START + codec,
                             constants.NOBLOCK | constants.SNDMORE)

        def dump(file):
            pickle.dump(message, file,
                        protocol=(4 if sys.version_info > (3, 4)
                                  else sys.version_info[0]))

        def send_pickle_end_marker():
            self.socket.send(
                ZmqConnection.PICKLE_END,
                constants.NOBLOCK | (constants.SNDMORE if not last else 0))

        def send_to_socket():
            send_pickle_beg_marker(codec)
            pickler = ZmqConnection.Pickler(self.socket, codec[0])
            dump(pickler)
            pickler.flush()
            send_pickle_end_marker()
            return pickler.size

        if io is None:
            return send_to_socket()
        else:
            try:
                initial_pos = io.tell()
                dump(io)
                new_pos = io.tell()
                send_pickle_beg_marker(b'\x00')
                io.seek(initial_pos)
                self.socket.send(pickle.dumps(io),
                                 constants.NOBLOCK | constants.SNDMORE)
                io.seek(new_pos)
                send_pickle_end_marker()
                return new_pos - initial_pos
            except ValueError:
                send_to_socket()
                raise ZmqConnection.IOOverflow()

    def messageReceived(self, message):
        """
        Called when complete message is received.

        Not implemented in :class:`ZmqConnection`, should be overridden to
        handle incoming messages.

        :param message: message data
        """
        raise NotImplementedError(self)

    def messageHeaderReceived(self, header):
        pass

    def _connectOrBind(self, endpoints):
        """
        Connect and/or bind socket to endpoints.
        """
        rnd_vals = []
        for endpoint in endpoints:
            if endpoint.type == ZmqEndpointType.connect:
                self.debug("Connecting to %s...", endpoint)
                self.socket.connect(endpoint.address)
            elif endpoint.type == ZmqEndpointType.bind:
                self.debug("Binding to %s...", endpoint)
                if endpoint.address.startswith("rndtcp://") or \
                   endpoint.address.startswith("rndepgm://"):
                    try:
                        endpos = endpoint.address.find("://") + 3
                        proto = endpoint.address[3:endpos]
                        splitted = endpoint.address[endpos:].split(':')
                        min_port, max_port, max_tries = splitted[-3:]
                        addr = ":".join(splitted[:-3])
                    except ValueError:
                        raise from_none(ValueError("Failed to parse %s" %
                                                   endpoint.address))
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
