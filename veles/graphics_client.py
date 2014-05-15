"""
Created on Jan 31, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
import datetime
import errno
import logging
import os
import platform
import signal
from six.moves import cPickle as pickle
import socket
import subprocess
import sys
import threading
import tornado.ioloop as ioloop
from twisted.internet import reactor
import zmq
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from veles.external.txzmq import ZmqConnection, ZmqEndpoint

from veles.config import root
from veles.logger import Logger


class ZmqSubscriber(ZmqConnection):

    socketType = zmq.SUB

    def __init__(self, graphics, *args, **kwargs):
        super(ZmqSubscriber, self).__init__(*args, **kwargs)
        self.socket.set(zmq.SUBSCRIBE, b'graphics')
        self.graphics = graphics

    def messageReceived(self, message):
        self.graphics.update(pickle.loads(memoryview(
            message[0])[len('graphics'):]))


class GraphicsClient(Logger):
    """ Class handling all interaction with the main graphics window.
    """

    ui_update_interval = 0.01

    def __init__(self, back, *endpoints, **kwargs):
        super(GraphicsClient, self).__init__()
        webagg_fifo = kwargs.get("webagg_fifo")
        self.back = back
        if self.back == "WebAgg":
            self._webagg_port = 0
        zmq_endpoints = []
        for ep in endpoints:
            zmq_endpoints.append(ZmqEndpoint("connect", ep))
        self.zmq_connection = ZmqSubscriber(self, zmq_endpoints)
        self._lock = threading.Lock()
        self._started = False
        self._shutted_down = False
        self.webagg_fifo = webagg_fifo
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
        try:
            if self._shutted_down:
                return
            self._started = True
            import matplotlib
            matplotlib.use(self.back)
            import matplotlib.cm as cm
            import matplotlib.lines as lines
            import matplotlib.patches as patches
            import matplotlib.pyplot as pp
            import veles.plotting_units  # do not remove
            try:
                import veles.znicz.nn_plotting_units  # do not remove
            except ImportError:
                self.warning("Failed to import veles.znicz.nn_plotting_units")
            pp.ion()
            self.matplotlib = matplotlib
            self.cm = cm
            self.lines = lines
            self.patches = patches
            self.pp = pp
            self.info("Graphics client is running in process %d", os.getpid())
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
                    free_port = root.common.matplotlib_webagg_port - 1
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
                        reactor.callLater(0, self._write_webagg_port, fifo)
        except:
            self._lock.release()
            raise
        if pp.get_backend() != "WebAgg":
            reactor.callLater(0, self._lock.release)
            reactor.run()
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
            reactor.callLater(0, self._write_webagg_port, fifo)
        else:
            self.debug("Wrote the WebAgg port to pipe")
            os.close(fifo)

    def update(self, plotter):
        """Processes one plotting event.
        """
        if plotter is not None:
            plotter.matplotlib = self.matplotlib
            plotter.cm = self.cm
            plotter.lines = self.lines
            plotter.patches = self.patches
            plotter.pp = self.pp
            plotter.show_figure = self.show_figure
            if self._pdf_trigger:
                reactor.callLater(0, self._save_pdf, plotter)
            else:
                reactor.callLater(0, plotter.redraw)
        else:
            self.debug("Received the command to terminate")
            self.shutdown()

    def show_figure(self, figure):
        if self.pp.get_backend() != "WebAgg":
            figure.show()
        else:
            with self.condition:
                self.condition.notify()

    def shutdown(self, urgent=False):
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
            if not urgent:
                reactor.stop()
            else:
                reactor.crash()
            # Not strictly necessary, but prevents from DoS
            self.zmq_connection.shutdown()

    def _save_pdf(self, plotter):
        with self._pdf_lock:
            figure = plotter.redraw()
            if plotter.id in self._pdf_units_served:
                self._pdf_trigger = False
                self._pdf_pages.close()
                self._pdf_pages = None
                self._pdf_units_served.clear()
                self._pdf_unit_chains.clear()
                self.info("Finished writing PDF %s" % self._pdf_file_name)
                system = platform.system()
                if system == "Windows":
                    os.startfile(self._pdf_file_name)
                elif system == "Linux":
                    subprocess.Popen(["xdg-open", self._pdf_file_name])
                self._pdf_file_name = None
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
                    (now.strftime('%Y_%m_%d_%H_%M_%S')))
                self.debug("Saving figures to %s...", self._pdf_file_name)
                import matplotlib.backends.backend_pdf as backend_pdf
                self._pdf_pages = backend_pdf.PdfPages(self._pdf_file_name)
            self._pdf_units_served.add(plotter.id)
            if getattr(plotter, "clear_plot", False):
                self._pdf_unit_chains.add(plotter.name)
            elif (not plotter.name in self._pdf_unit_chains or
                  getattr(plotter, "redraw_plot", False)):
                self._pdf_pages.savefig(figure)

    def _sigint_handler(self, sign, frame):
        self.shutdown(True)
        try:
            self._sigint_initial(sign, frame)
        except KeyboardInterrupt:
            self.critical("KeyboardInterrupt")
            reactor.stop()

    def _sigusr2_handler(self, sign, frame):
        self._pdf_trigger = True


if __name__ == "__main__":
    Logger.setup(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend", nargs='?',
                        default=root.common.matplotlib_backend,
                        help="Matplotlib drawing backend.")
    parser.add_argument("-e", "--endpoint", required=True,
                        help="ZeroMQ endpoint to receive updates from.")
    parser.add_argument("--webagg-discovery-fifo", nargs='?',
                        default=None, help="Matplotlib drawing backend.")
    cmdargs = parser.parse_args()
    client = GraphicsClient(cmdargs.backend, cmdargs.endpoint,
                            webagg_fifo=cmdargs.webagg_discovery_fifo)
    if cmdargs.backend == "WebAgg":
        client_thread = threading.Thread(target=client.run)
        client_thread.start()
        reactor.run()
        client_thread.join()
    else:
        client.run()
