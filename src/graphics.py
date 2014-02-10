"""
Created on Jan 31, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from PyQt4 import QtGui, QtCore
import logging
import matplotlib
import queue
import time
import tkinter

import matplotlib.pyplot as pp
import multiprocessing as mp


class Graphics(object):
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

    @staticmethod
    def initialize():
        if not Graphics.process:
            pp.ion()
            Graphics.event_queue = mp.Queue()
            Graphics.process = mp.Process(target=Graphics.server_entry)
            Graphics.process.start()

    @staticmethod
    def enqueue(obj):
        if not Graphics.process:
            raise RuntimeError("Graphics is not initialized")
        Graphics.event_queue.put_nowait(obj)

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
        self.exiting = False
        self.showed = False
        self.root = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        self.logger.info("Server is running")
        if pp.get_backend() == "TkAgg":
            self.root = tkinter.Tk()
            self.root.withdraw()
            self.root.after(100, self.update)
            tkinter.mainloop()
        elif pp.get_backend() == "Qt4Agg":
            self.root = QtGui.QApplication([])
            self.timer = QtCore.QTimer(self.root)
            self.timer.timeout.connect(self.update)
            self.timer.start(100)
            self.root.exec_()
        elif pp.get_backend() == "WebAgg":
            matplotlib.rcParams['webagg.port'] = 8888
            matplotlib.rcParams['webagg.open_in_browser'] = 'False'
            while not self.exiting:
                self.update()
                time.sleep(0.1)

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
                if pp.get_backend() == "WebAgg" and not self.showed:
                    self.showed = True
                    pp.show()
        except queue.Empty:
            pass
        if pp.get_backend() == "TkAgg":
            if not self.exiting:
                self.root.after(100, self.update)
            else:
                self.logger.debug("Terminating the main loop.")
                self.root.destroy()
        if pp.get_backend() == "Qt4Agg" and self.exiting:
            self.timer.stop()
            self.logger.debug("Terminating the main loop.")
            self.root.quit()
