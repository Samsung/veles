"""
Created on Jan 21, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import argparse
import copy
import logging
import signal
import six
from six.moves import queue, zip
from six import add_metaclass
import sys
import threading
from traceback import print_stack
import types
from twisted.python import threadpool

import veles.logger as logger
from veles.cmdline import CommandLineArgumentsRegistry


sysexit_initial = None


@add_metaclass(CommandLineArgumentsRegistry)
class ThreadPool(threadpool.ThreadPool, logger.Logger):
    """
    Pool of threads.
    """

    sigint_initial = None
    pools = []
    _manhole = None

    def __init__(self, minthreads=2, maxthreads=1024, queue_size=2048,
                 name=None):
        """
        Creates a new thread pool and starts it.
        """
        if six.PY3:
            super(ThreadPool, self).__init__(
                minthreads=minthreads, maxthreads=maxthreads, name=name)
        else:
            threadpool.ThreadPool.__init__(
                self, minthreads=minthreads, maxthreads=maxthreads, name=name)
        logger.Logger.__init__(self)
        self.q = queue.Queue(queue_size)
        self.start()
        self.on_shutdowns = []
        self.silent = False
        self._shutting_down = False
        self._shutting_down_lock_ = threading.Lock()
        self._not_paused = threading.Event()
        self._not_paused.set()

        if not ThreadPool.pools:
            global sysexit_initial
            sysexit_initial = sys.exit
            sys.exit = ThreadPool.exit
            ThreadPool.sigint_initial = \
                signal.signal(signal.SIGINT, ThreadPool.sigint_handler)
            if not ThreadPool.manhole:
                signal.signal(signal.SIGUSR1, ThreadPool.sigusr1_handler)
            else:
                from veles.external import manhole
                manhole.install(patch_fork=False)
        ThreadPool.pools.append(self)

    def __del__(self):
        if not self.joined:
            self.shutdown(False, True)

    @staticmethod
    def init_parser(**kwargs):
        parser = kwargs.get("parser", argparse.ArgumentParser())
        parser.add_argument("--manhole", default=False,
                            help="run an embedded interactive shell "
                            "accessible though a UNIX socket",
                            action='store_true')
        return parser

    @property
    @staticmethod
    def manhole():
        if ThreadPool._manhole is None:
            parser = ThreadPool.init_parser()
            args, _ = parser.parse_known_args()
            ThreadPool._manhole = args.manhole
        return ThreadPool._manhole

    def callInThreadWithCallback(self, onResult, func, *args, **kw):
        self._not_paused.wait()
        with self._shutting_down_lock_:
            if self._shutting_down or not self.started:
                return
            threadpool.ThreadPool.callInThreadWithCallback(self, onResult,
                                                           func, *args, **kw)

    def pause(self):
        self._not_paused.clear()
        self.info("ThreadPool with %d threads has been suspended",
                  len(self.threads))

    def resume(self):
        self._not_paused.set()
        self.info("ThreadPool with %d threads has been resumed",
                  len(self.threads))

    @property
    def paused(self):
        return not self._not_paused.is_set()

    def register_on_shutdown(self, func):
        """
        Adds the specified function to the list of callbacks which are
        executed before shutting down the thread pool.
        It is useful when an infinite event loop is executed in a separate
        thread and a graceful shutdown is desired. Then on_shutdown() function
        shall terminate that loop using the corresponding foreign API.
        """
        self.on_shutdowns.append(func)

    @staticmethod
    def _put(self, item):
        """
        Private method used by shutdown() to redefine Queue's _put().
        """
        self.queue.appendleft(item)

    def shutdown(self, execute_remaining=True, force=False, timeout=0.25):
        """Safely brings thread pool down.
        """
        with self._shutting_down_lock_:
            if self not in ThreadPool.pools or self._shutting_down:
                return
            self._shutting_down = True
        self._not_paused.set()
        sdl = len(self.on_shutdowns)
        self.debug("Running shutdown-ers")
        for on_shutdown, ind in zip(self.on_shutdowns, range(sdl)):
            self.debug("%d/%d - %s", ind, sdl, str(on_shutdown))
            on_shutdown()
        self.debug("Requesting threads to quit")
        del self.on_shutdowns[:]
        self.joined = True
        self.started = False
        threads = copy.copy(self.threads)
        if not execute_remaining:
            self.q._put = types.MethodType(ThreadPool._put, self.q)
        while self.workers:
            self.q.put(threadpool.WorkerStop)
            self.workers -= 1
        self.debug("Joining threads")
        for thread in threads:
            if not force:
                thread.join()
            else:
                thread.join(timeout)
                if thread.is_alive():
                    if not self.silent:
                        self.warning("Stack trace of probably deadlocked #%d:",
                                     thread.ident)
                        print_stack(sys._current_frames()[thread.ident],
                                    file=sys.stdout)
                    self.force_thread_to_stop(thread)
                    if not self.silent:
                        self.warning(
                            "Failed to join with thread #%d since the  timeout"
                            " (%.2f sec) was exceeded.%s",
                            thread.ident, timeout, " It was killed."
                            if ThreadPool.thread_can_be_forced_to_stop(thread)
                            else " It was not killed due to the lack of _stop "
                            "in Thread class from Python's stdlib.")
        ThreadPool.pools.remove(self)
        if not len(ThreadPool.pools):
            global sysexit_initial
            sys.exit = sysexit_initial
            # if ThreadPool.manhole:
            #    manhole.
        self.debug("%s was shutted down", repr(self))
        self._shutting_down = False

    @staticmethod
    def thread_can_be_forced_to_stop(thread):
        return hasattr(thread, "_stop") and callable(thread._stop) and \
            thread != threading.main_thread() and thread.is_alive()

    def force_thread_to_stop(self, thread):
        if ThreadPool.thread_can_be_forced_to_stop(thread):
            try:
                thread._stop()
            except:
                if not self.silent:
                    self.warning("Failed to kill %s", str(thread))

    @staticmethod
    def shutdown_pools(execute_remaining=True, force=False, timeout=0.25):
        """
        Private method to shut down all the pools.
        """
        pools = copy.copy(ThreadPool.pools)
        for pool in pools:
            pool.shutdown(execute_remaining, force, timeout)

    @staticmethod
    def exit(retcode=0):
        """
        Terminates the running program safely.
        """
        ThreadPool.shutdown_pools()
        ThreadPool.debug_deadlocks()
        sys.exit(retcode)

    @staticmethod
    def sigint_handler(sign, frame):
        """
        Private method - handler for SIGINT.
        """
        ThreadPool.shutdown_pools(execute_remaining=False, force=True)
        try:
            ThreadPool.sigint_initial(sign, frame)
        except KeyboardInterrupt:
            logging.getLogger("ThreadPool").critical("KeyboardInterrupt")

    @staticmethod
    def sigusr1_handler(sign, frame):
        """
        Private method - handler for SIGUSR1.
        """
        print("SIGUSR1 was received, dumping current frames...")
        ThreadPool.print_thread_stacks()

    @staticmethod
    def print_thread_stacks():
        if not hasattr(sys, "_current_frames"):
            print("Threads' stacks printing is not implemented for this "
                  "Python interpreter")
            return
        tmap = {thr.ident: thr.name for thr in threading.enumerate()}
        for tid, stack in sys._current_frames().items():
            print("-" * 80)
            print("Thread #%d (%s):" % (tid, tmap.get(tid, "<unknown>")))
            print_stack(stack, file=sys.stdout)

    @staticmethod
    def debug_deadlocks():
        if threading.active_count() > 1:
            if threading.active_count() == 2:
                for thread in threading.enumerate():
                    if thread.name.startswith('timeout'):
                        # veles.tests.timeout registers atexit
                        return
            logging.warning("There are currently more than 1 threads still "
                            "running. A deadlock is likely to happen.\n%s",
                            str(threading.enumerate())
                            if hasattr(threading, "_active")
                            else "<unable to list active threads>")
            ThreadPool.print_thread_stacks()
