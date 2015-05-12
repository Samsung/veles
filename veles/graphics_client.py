# -*- coding: utf-8 -*-
"""
.. invisible:
    _   _ _____ _     _____ _____
   | | | |  ___| |   |  ___/  ___|
   | | | | |__ | |   | |__ \ `--.
   | | | |  __|| |   |  __| `--. \
   \ \_/ / |___| |___| |___/\__/ /
    \___/\____/\_____|____/\____/

Created on Jan 31, 2014

Contains GraphicsClient cllass, which handling all interaction with the main
graphics window

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
import datetime
import errno
import gc
import logging
from importlib import import_module
import os
import signal
import socket
import threading

import snappy
import tornado.ioloop as ioloop
from twisted.internet.error import ReactorNotRunning
from twisted.internet import reactor
import zmq

from veles.config import root
from veles.txzmq import ZmqConnection, ZmqEndpoint
from veles.iplotter import IPlotter
from veles.logger import Logger
from veles.pickle2 import pickle, setup_pickle_debug


class ZmqSubscriber(ZmqConnection):

    socketType = zmq.SUB

    def __init__(self, graphics, *args, **kwargs):
        super(ZmqSubscriber, self).__init__(*args, **kwargs)
        self.socket.set(zmq.SUBSCRIBE, b'graphics')
        self.graphics = graphics

    def messageReceived(self, message):
        self.graphics.debug("Received %d bytes", len(message[0]))
        raw_data = snappy.decompress(message[0][len('graphics'):])
        obj = pickle.loads(raw_data)
        self.graphics.update(obj, raw_data)


class GraphicsClient(Logger):
    """ Class handling all interaction with the main graphics window.
    """

    ui_update_interval = 0.01
    gc_limit = 10

    def __init__(self, backend, *endpoints, **kwargs):
        super(GraphicsClient, self).__init__()
        webagg_fifo = kwargs.get("webagg_fifo")
        self.backend = backend
        if self.backend == "WebAgg":
            self._webagg_port = 0
        zmq_endpoints = []
        for ep in endpoints:
            zmq_endpoints.append(ZmqEndpoint("connect", ep))
        self.zmq_connection = ZmqSubscriber(self, zmq_endpoints)
        self._lock = threading.Lock()
        self._started = False
        self._shutted_down = False
        self.webagg_fifo = webagg_fifo
        self._gc_counter = 0
        self._dump_dir = kwargs.get("dump_dir")
        self._pdf_lock = threading.Lock()
        self._pdf_trigger = False
        self._pdf_pages = None
        self._pdf_file_name = None
        self._pdf_units_served = set()
        self._pdf_unit_chains = set()
        self._sigint_initial = signal.signal(signal.SIGINT,
                                             self._sigint_handler)
        self._sigusr2_initial = signal.signal(signal.SIGUSR2,
                                              self._sigusr2_handler)

    def __del__(self):
        signal.signal(signal.SIGINT, self._sigint_initial)

    def run(self):
        """Creates and runs main graphics window.
        """
        self._lock.acquire()
        if self.backend == "no":
            self._run()
            return
        Plotter = import_module("veles.plotter").Plotter
        try:
            if self._shutted_down:
                return
            self._started = True
            import matplotlib
            if self.backend:
                matplotlib.use(self.backend)
            try:
                self.pkgs = Plotter.import_matplotlib()
            except ImportError:
                self.warning("%s backend is not loadable, falling back to "
                             "WebAgg")
                matplotlib.use("WebAgg", force=True)
                try:
                    self.pkgs = Plotter.import_matplotlib()
                except ImportError:
                    self.exception("Failed to load WebAgg matplotlib backend")
                    return
            self.pp = pp = self.pkgs["pp"]
            pp.ion()
            if pp.get_backend() == "TkAgg":
                from six.moves import tkinter
                self.root = tkinter.Tk()
                self.root.withdraw()
                reactor.callLater(GraphicsClient.ui_update_interval,
                                  self._process_tk_events)
                # tkinter.mainloop()
            elif pp.get_backend() == "Qt4Agg":
                import PyQt4
                self.root = PyQt4.QtGui.QApplication([])
                reactor.callLater(GraphicsClient.ui_update_interval,
                                  self._process_qt_events)
                # self.root.exec_()
            elif pp.get_backend() == "WxAgg":
                import wx
                self.root = wx.PySimpleApp()
                reactor.callLater(GraphicsClient.ui_update_interval,
                                  self._process_wx_events)
                # self.root.MainLoop()
            elif pp.get_backend() == "WebAgg":
                self.condition = threading.Condition()
                with self.condition:
                    self.condition.wait()
                    free_port = root.common.graphics.matplotlib.webagg_port - 1
                    result = 0
                    while result == 0:
                        free_port += 1
                        sock = socket.socket(socket.AF_INET,
                                             socket.SOCK_STREAM)
                        result = sock.connect_ex(("localhost", free_port))
                        sock.close()
                    self._webagg_port = free_port
                    matplotlib.rcParams['webagg.port'] = free_port
                    matplotlib.rcParams['webagg.open_in_browser'] = 'False'
                    self.info("Launching WebAgg instance on port %d",
                              free_port)
                    if self.webagg_fifo is not None:
                        fifo = os.open(self.webagg_fifo,
                                       os.O_WRONLY | os.O_NONBLOCK)
                        self._webagg_port_bytes = \
                            str(self._webagg_port).encode()
                        reactor.callWhenRunning(self._write_webagg_port, fifo)
        except:
            self._lock.release()
            raise
        self._run()

    def update(self, plotter, raw_data):
        """Processes one plotting event.
        """
        if plotter is not None:
            if self._dump_dir:
                file_name = os.path.join(self._dump_dir, "%s_%s.pickle" % (
                    plotter.name.replace(" ", "_"),
                    datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
                with open(file_name, "wb") as fout:
                    fout.write(raw_data)
                self.info("Wrote %s", file_name)
            del raw_data
            self._gc_counter += 1
            if self._gc_counter >= GraphicsClient.gc_limit:
                gc.collect()
                self._gc_counter = 0

            if self.backend == "no":
                return
            plotter.set_matplotlib(self.pkgs)
            plotter.show_figure = self.show_figure
            if not IPlotter.providedBy(plotter):
                self.warning("%s does not provide IPlotter interface", plotter)
                return
            try:
                plotter.verify_interface(IPlotter)
            except:
                self.exception("Plotter %s is not fully implemented, skipped",
                               plotter)
                return
            if self._pdf_trigger or self.backend == "pdf":
                reactor.callFromThread(self._save_pdf, plotter)
            else:
                reactor.callFromThread(plotter.redraw)
        else:
            self.debug("Received the command to terminate")
            self.shutdown()

    def show_figure(self, figure):
        if self.pp.get_backend() != "WebAgg":
            figure.show()
        else:
            with self.condition:
                self.condition.notify()

    def shutdown(self):
        with self._lock:
            if not self._started or self._shutted_down:
                return
            self.info("Shutting down")
            self._shutted_down = True
            if self.pp.get_backend() == "TkAgg":
                self.root.destroy()
            elif self.pp.get_backend() == "Qt4Agg":
                self.root.quit()
            elif self.pp.get_backend() == "WxAgg":
                self.root.ExitMainLoop()
            elif self.pp.get_backend() == "WebAgg":
                ioloop.IOLoop.instance().stop()
            try:
                reactor.stop()
            except ReactorNotRunning:
                pass
            # Not strictly necessary, but prevents from DoS
            self.zmq_connection.shutdown()

    def _run(self):
        self.info("Graphics client is running in process %d", os.getpid())
        if self.backend == "no" or self.pp.get_backend() != "WebAgg":
            reactor.callWhenRunning(self._lock.release)
            try:
                reactor.run()
            except ReactorNotRunning:
                pass
        else:
            ioloop.IOLoop.instance().add_callback(self._lock.release)
            self.pp.show()
        self.info("Finished")

    def _process_qt_events(self):
        self.root.processEvents()
        reactor.callLater(GraphicsClient.ui_update_interval,
                          self._process_qt_events)

    def _process_tk_events(self):
        self.root.update()
        reactor.callLater(GraphicsClient.ui_update_interval,
                          self._process_tk_events)

    def _process_wx_events(self):
        self.root.ProcessPendingEvents()
        reactor.callLater(GraphicsClient.ui_update_interval,
                          self._process_wx_events)

    def _write_webagg_port(self, fifo):
        try:
            written = os.write(fifo, self._webagg_port_bytes)
        except (OSError, IOError) as ioe:
            if ioe.args[0] in (errno.EAGAIN, errno.EINTR):
                written = 0
        if written != len(self._webagg_port_bytes):
            reactor.callWhenRunning(self._write_webagg_port, fifo)
        else:
            self.debug("Wrote the WebAgg port to pipe")
            os.close(fifo)

    def _save_pdf(self, plotter):
        with self._pdf_lock:
            figure = plotter.redraw()
            if plotter.id in self._pdf_units_served:
                from veles.portable import show_file

                self._pdf_trigger = False
                self._pdf_pages.close()
                self._pdf_pages = None
                self._pdf_units_served.clear()
                self._pdf_unit_chains.clear()
                self.info("Finished writing PDF %s", self._pdf_file_name)
                show_file(self._pdf_file_name)
                self._pdf_file_name = None
                if self.backend != "pdf":
                    return
            if self._pdf_pages is None:
                now = datetime.datetime.now()
                out_dir = os.path.join(root.common.cache_dir, "plots")
                try:
                    os.makedirs(out_dir, mode=0o775)
                except OSError:
                    pass
                self._pdf_file_name = os.path.join(
                    root.common.cache_dir, "plots/veles_%s.pdf" %
                    (now.strftime('%Y-%m-%d_%H:%M:%S')))
                self.debug("Saving figures to %s...", self._pdf_file_name)
                import matplotlib.backends.backend_pdf as backend_pdf
                self._pdf_pages = backend_pdf.PdfPages(self._pdf_file_name)
            self._pdf_units_served.add(plotter.id)
            if getattr(plotter, "clear_plot", False):
                self._pdf_unit_chains.add(plotter.name)
            elif (plotter.name not in self._pdf_unit_chains or
                  getattr(plotter, "redraw_plot", False)):
                self._pdf_pages.savefig(figure)

    def _sigint_handler(self, sign, frame):
        self.shutdown()
        try:
            self._sigint_initial(sign, frame)
        except KeyboardInterrupt:
            self.critical("KeyboardInterrupt")

            def stop():
                try:
                    reactor.stop()
                except ReactorNotRunning:
                    pass

            reactor.callWhenRunning(stop)

    def _sigusr2_handler(self, sign, frame):
        self.info("Activated PDF mode...")
        self._pdf_trigger = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend", nargs='?',
                        default=root.common.graphics.matplotlib.backend,
                        help="Matplotlib drawing backend. \"no\" value disable"
                             "s any real plotting (useful with --dump).")
    parser.add_argument("-e", "--endpoint", required=True,
                        help="ZeroMQ endpoint to receive updates from.")
    parser.add_argument("--webagg-discovery-fifo", nargs='?',
                        default=None, help="Matplotlib drawing backend.")
    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}
    parser.add_argument("-v", "--verbose", type=str, default="info",
                        choices=LOG_LEVEL_MAP.keys(),
                        help="set verbosity level [default: %(default)s]")
    parser.add_argument("-d", "--dump", type=str, default="",
                        help="Dump incoming messages to this directory.")
    cmdargs = parser.parse_args()

    log_level = LOG_LEVEL_MAP[cmdargs.verbose]
    Logger.setup_logging(level=log_level)
    if log_level == logging.DEBUG:
        setup_pickle_debug()

    client = GraphicsClient(cmdargs.backend, cmdargs.endpoint,
                            webagg_fifo=cmdargs.webagg_discovery_fifo,
                            dump_dir=cmdargs.dump)
    if log_level == logging.DEBUG:
        client.debug("Activated pickle debugging")
    if cmdargs.backend == "WebAgg":
        client_thread = threading.Thread(target=client.run)
        client_thread.start()
        reactor.run()
        client_thread.join()
    else:
        client.run()


if __name__ == "__main__":
    main()
