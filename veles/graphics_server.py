# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 7, 2014

Graphics server which uses ZeroMQ PUB socket to publish updates.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import argparse
import errno
from fcntl import fcntl, F_GETFL, F_SETFL
from io import UnsupportedOperation
import os
import subprocess
import sys
from tempfile import mkdtemp
import threading
from select import select
import six
import snappy
from twisted.internet import reactor
import zmq

from six.moves import cPickle as pickle, zip
from veles.cmdline import CommandLineArgumentsRegistry
from veles.compat import from_none
from veles.config import root
from veles.txzmq import ZmqConnection, ZmqEndpoint
import veles.graphics_client as graphics_client
from veles.logger import Logger
from veles.network_common import interfaces
from veles.paths import __root__
from veles.timeit2 import timeit


class ZmqPublisher(ZmqConnection):
    socketType = zmq.PUB

    def send(self, message):
        super(ZmqPublisher, self).send(b'graphics' + message)


@six.add_metaclass(CommandLineArgumentsRegistry)
class GraphicsServer(Logger):
    """
    Graphics server which uses ZeroMQ PUB socket to publish updates.
    """

    class InitializationError(Exception):
        pass

    _instance = None
    _pair_fds = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GraphicsServer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, thread_pool=None):
        if self.initialized:
            return
        self.initialized = True
        assert thread_pool is not None, (
            "GraphicsServer was not previously initialized")
        super(GraphicsServer, self).__init__()
        parser = GraphicsServer.init_parser()
        args, _ = parser.parse_known_args(self.argv)
        self._debug_pickle = args.graphics_pickle_debug
        zmq_endpoints = [ZmqEndpoint("bind", "inproc://veles-plots"),
                         ZmqEndpoint("bind", "rndipc://veles-ipc-plots-:")]
        ifaces = []
        for iface, _ in interfaces():
            if iface in root.common.graphics.blacklisted_ifaces:
                continue
            ifaces.append(iface)
            zmq_endpoints.append(ZmqEndpoint(
                "bind", "rndepgm://%s;%s:1024:65535:1" %
                        (iface, root.common.graphics.multicast_address)))
        self.debug("Trying to bind to %s...", zmq_endpoints)

        try:
            self.zmq_connection, btime = timeit(ZmqPublisher, zmq_endpoints)
        except zmq.error.ZMQError:
            self.exception("Failed to bind to %s", zmq_endpoints)
            raise from_none(GraphicsServer.InitializationError())

        # Important! Save the bound method to variable to avoid dead weak refs
        # See http://stackoverflow.com/questions/19443440/weak-reference-to-python-class-method  # nopep8
        self._shutdown_ = self.shutdown
        thread_pool.register_on_shutdown(self._shutdown_)

        # tmpfn, *ports = self.zmq_connection.rnd_vals
        tmpfn = self.zmq_connection.rnd_vals[0]
        ports = self.zmq_connection.rnd_vals[1:]

        self.endpoints = {"inproc": "inproc://veles-plots",
                          "ipc": "ipc://" + tmpfn,
                          "epgm": []}
        for port, iface in zip(ports, ifaces):
            self.endpoints["epgm"].append(
                "epgm://%s;%s:%d" %
                (iface, root.common.graphics.multicast_address, port))
        self.info("Publishing to %s", "; ".join([self.endpoints["inproc"],
                                                 self.endpoints["ipc"]] +
                                                self.endpoints["epgm"]))
        if btime > 1:
            self.warning(
                "EPGM bind took %d seconds - consider adding offending "
                "interfaces to root.common.graphics.blacklisted_ifaces or "
                "completely disabling graphics (-p '').",
                int(btime))

    @staticmethod
    def init_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--graphics-pickle-debug", default=False,
                            action="store_true",
                            help="Save plotter object trees during server "
                            "send.")
        return parser

    def enqueue(self, obj):
        data = pickle.dumps(obj)
        if getattr(self, "_debug_pickle", False):
            import objgraph
            restored = pickle.loads(data)
            objgraph.show_refs(restored, too_many=40)
        data = snappy.compress(data)
        self.debug("Broadcasting %d bytes" % len(data))
        zmq_connection = getattr(self, "zmq_connection")
        if zmq_connection is not None:
            zmq_connection.send(data)

    def shutdown(self):
        self.debug("Shutting down")
        if hasattr(self, "fifo_shutting_down"):
            self.fifo_shutting_down = True
        self.enqueue(None)
        if hasattr(self, "fifo_thread") and self.fifo_thread.is_alive():
            self.fifo_thread.join()

    @staticmethod
    def launch(thread_pool, backend, webagg_callback=None, only_server=False):
        server = GraphicsServer(thread_pool)
        if only_server:
            return server, None
        if six.PY3:
            python = "python3"
        else:
            python = "python"
        args = ["env", python, graphics_client.__file__,
                "--backend", backend, "--endpoint", server.endpoints["ipc"]]
        if backend == "WebAgg" and \
           webagg_callback is not None:
            tmpdir = mkdtemp(prefix="veles-graphics")
            tmpfn = os.path.join(tmpdir, "comm")
            os.mkfifo(tmpfn)
            fifo = os.open(tmpfn, os.O_RDONLY | os.O_NONBLOCK)
            reactor.callLater(0, GraphicsServer._read_webagg_port,
                              fifo, tmpfn, tmpdir, webagg_callback)
            args.append("--webagg-discovery-fifo")
            args.append(tmpfn)
        env = dict(os.environ)
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = __root__
        else:
            env['PYTHONPATH'] += ':' + __root__
        streams = {"stdout": None, "stderr": None}
        try:
            for key in streams:
                stream = getattr(sys, key)
                stream.fileno()
                streams[key] = stream
            pipe = False
        except (UnsupportedOperation, AttributeError) as e:
            server.debug("fileno() check failed: %s", e)
            for key in streams:
                streams[key] = subprocess.PIPE
            pipe = True
        client = subprocess.Popen(args, env=env, **streams)
        if pipe:
            for stream in client.stdout, client.stderr:
                fcntl(stream, F_SETFL, fcntl(stream, F_GETFL) | os.O_NONBLOCK)
            server.fifo_shutting_down = False
            server.fifo_thread = threading.Thread(
                target=server.drain_fifo, args=(client,))
            server.fifo_thread.start()
            server.debug("Started the stdout / stderr pipe thread")
        return server, client

    def drain_fifo(self, client):
        while not self.fifo_shutting_down:
            rs, _, _ = select((client.stdout, client.stderr), tuple(), tuple())
            for stream in rs:
                out = sys.stdout if stream is client.stdout else sys.stderr
                out.write(stream.read().decode("utf-8"))
                out.flush()

    @staticmethod
    def _read_webagg_port(fifo, tmpfn, tmpdir, webagg_callback):
        try:
            output = os.read(fifo, 8)
        except (OSError, IOError) as ioe:
            if ioe.args[0] in (errno.EAGAIN, errno.EINTR):
                output = None
        if not output:
            reactor.callLater(0.1, GraphicsServer._read_webagg_port,
                              fifo, tmpfn, tmpdir, webagg_callback)
        else:
            os.close(fifo)
            os.unlink(tmpfn)
            os.rmdir(tmpdir)
            if webagg_callback is not None:
                webagg_callback(int(output))
