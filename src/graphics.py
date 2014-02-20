"""
Created on Jan 31, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import multiprocessing
import os
import socket
import queue
import time
import threading
import tornado.ioloop as ioloop

import config
import logger


class Graphics(logger.Logger):
    """ Class handling all interaction with main graphics window
        NOTE: This class should be created ONLY within one thread
        (preferably main)

    Attributes:
        _instance: instance of Graphics class. Used for implementing
            Singleton pattern for this class.
        root: TKinter graphics root.
        event_queue: Queue of all pending changes created by other threads.
        run_lock: Lock to determine whether graphics window is running
        registered_plotters: List of registered plotters
        is_initialized: whether this class was already initialized.
    """

    _instance = None
    event_queue = None
    process = None
    interval = 0.1  # secs in event loop
    matplotlib_webagg_listened_port = multiprocessing.Value(
        'i', config.matplotlib_webagg_port)

    @staticmethod
    def initialize():
        if not Graphics.process:
            # cache only 20 drawing events
            Graphics.event_queue = multiprocessing.Queue(20)
            """ TODO(v.markovtsev): solve the problem with matplotlib, ssh and
            multiprocessing - hangs on figure.show()
            """
            import matplotlib
            mplver = matplotlib.__version__
            del(matplotlib)
            if mplver == "1.4.x":
                Graphics.process = threading.Thread(
                    target=Graphics.server_entry)
            else:
                Graphics.process = multiprocessing.Process(
                    target=Graphics.server_entry)
            Graphics.process.start()

    @staticmethod
    def enqueue(obj):
        if not Graphics.process:
            raise RuntimeError("Graphics is not initialized")
        try:
            Graphics.event_queue.put_nowait(obj)
        except queue.Full:
            pass

    @staticmethod
    def server_entry():
        Graphics().run()

    @staticmethod
    def shutdown():
        Graphics.enqueue(None)
        if Graphics.process:
            Graphics.process.join()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Graphics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        super(Graphics, self).__init__()
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        if not Graphics.process:
            raise RuntimeError("Graphics server must be launched before "
                               "the initialization")
        self.exiting = False
        self.showed = False
        self.root = None
        self.webagg_port = 0

    def run(self):
        import matplotlib
        matplotlib.use(config.matplotlib_backend)
        import matplotlib.cm as cm
        import matplotlib.lines as lines
        import matplotlib.patches as patches
        import matplotlib.pyplot as pp
        pp.ion()
        self.matplotlib = matplotlib
        self.cm = cm
        self.lines = lines
        self.patches = patches
        self.pp = pp
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        self.info("Graphics server is running in process %d", os.getpid())
        if pp.get_backend() == "TkAgg":
            import tkinter
            self.root = tkinter.Tk()
            self.root.withdraw()
            self.root.after(100, self.update)
            tkinter.mainloop()
        elif pp.get_backend() == "Qt4Agg":
            from PyQt4 import QtGui, QtCore
            self.root = QtGui.QApplication([])
            self.timer = QtCore.QTimer(self.root)
            self.timer.timeout.connect(self.update)
            self.timer.start(Graphics.interval * 1000)
            self.root.exec_()
        elif pp.get_backend() == "WebAgg":
            free_port = config.matplotlib_webagg_port - 1
            result = 0
            while result == 0:
                free_port += 1
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    result = sock.connect_ex(("localhost", free_port))
            self.info("Will launch WebAgg instance on port %d", free_port)
            self.webagg_port = free_port
            Graphics.matplotlib_webagg_listened_port = free_port
            matplotlib.rcParams['webagg.port'] = free_port
            matplotlib.rcParams['webagg.open_in_browser'] = 'False'
            self.webagg_thread = threading.Thread(target=self._run_webagg)
            while not self.exiting:
                self.update()
                time.sleep(Graphics.interval)
            ioloop.IOLoop.instance().stop()
            self.webagg_thread.join()

    def _run_webagg(self):
        self.pp.show()

    def update(self):
        """Processes all events scheduled for plotting
        """
        try:
            processed = set()
            while True:
                plotter = self.event_queue.get_nowait()
                if not plotter:
                    self.exiting = True
                    break
                if plotter in processed:
                    continue
                processed.add(plotter)
                plotter.redraw()
                if self.pp.get_backend() == "WebAgg" and not self.showed:
                    self.showed = True
                    self.info("Starting WebAgg...")
                    self.webagg_thread.start()
        except queue.Empty:
            pass
        if self.pp.get_backend() == "TkAgg":
            if not self.exiting:
                self.root.after(Graphics.interval * 1000, self.update)
            else:
                self.debug("Terminating the main loop")
                self.root.destroy()
        if self.pp.get_backend() == "Qt4Agg" and self.exiting:
            self.timer.stop()
            self.debug("Terminating the main loop")
            self.root.quit()

    def show_figure(self, figure):
        if self.pp.get_backend() != "WebAgg":
            figure.show()
