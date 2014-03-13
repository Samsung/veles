"""
Created on Jan 31, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import os
import signal
from six.moves import cPickle as pickle
import socket
import sys
import threading
import tornado.ioloop as ioloop
from twisted.internet import reactor
from txzmq import ZmqConnection, ZmqEndpoint
import zmq

import config
from logger import Logger


class ZmqSubscriber(ZmqConnection):
    socketType = zmq.constants.SUB

    def __init__(self, graphics, *args, **kwargs):
        super(ZmqSubscriber, self).__init__(*args, **kwargs)
        self.socket.set(zmq.constants.SUBSCRIBE, b'')
        self.graphics = graphics

    def messageReceived(self, message):
        self.graphics.update(pickle.loads(message[0]))


class GraphicsClient(Logger):
    """ Class handling all interaction with the main graphics window.
    """

    ui_update_interval = 0.01

    def __init__(self, backend, *endpoints, webagg_fifo=None):
        super(GraphicsClient, self).__init__()
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
        self._sigint_initial = signal.signal(signal.SIGINT,
                                             self._sigint_handler)

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
            matplotlib.use(self.backend)
            import matplotlib.cm as cm
            import matplotlib.lines as lines
            import matplotlib.patches as patches
            import matplotlib.pyplot as pp
            import plotting_units  # important - do not remove
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
            elif pp.get_backend() == "WebAgg":
                self.condition = threading.Condition()
                with self.condition:
                    self.condition.wait()
                    free_port = config.matplotlib_webagg_port - 1
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
        if plotter != None:
            plotter.matplotlib = self.matplotlib
            plotter.cm = self.cm
            plotter.lines = self.lines
            plotter.patches = self.patches
            plotter.pp = self.pp
            plotter.show_figure = self.show_figure
            plotter.redraw()
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
            elif self.pp.get_backend() == "WebAgg":
                ioloop.IOLoop.instance().stop()
            if not urgent:
                reactor.stop()
            else:
                reactor.crash()
            # Not strictly necessary, but prevents from DoS
            self.zmq_connection.shutdown()

    def _sigint_handler(self, signal, frame):
        self.shutdown(True)
        self._sigint_initial(signal, frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backend, endpoint = sys.argv[1:3]
    client = GraphicsClient(backend, endpoint,
                            webagg_fifo=sys.argv[3]
                            if len(sys.argv) > 3 else None)
    if backend == "WebAgg":
        client_thread = threading.Thread(target=client.run)
        client_thread.start()
        reactor.run()
        client_thread.join()
    else:
        client.run()
