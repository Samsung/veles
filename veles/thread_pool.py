"""
Created on Jan 21, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import copy
import logging
import signal
import six
from six.moves import queue
from six.moves import zip
import sys
import threading
import traceback
import types
from twisted.python import threadpool

import veles.logger as logger


class ThreadPool(threadpool.ThreadPool, logger.Logger):
    """
    Pool of threads.
    """

    sysexit_initial = None
    sigint_initial = None
    pools = []

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
        self.shutting_down = False
        if not ThreadPool.pools:
            ThreadPool.sysexit_initial = sys.exit
            sys.exit = ThreadPool.exit
            ThreadPool.sigint_initial = \
                signal.signal(signal.SIGINT, ThreadPool.sigint_handler)
            signal.signal(signal.SIGUSR1, ThreadPool.sigusr1_handler)
        ThreadPool.pools.append(self)

    def __fini__(self):
        if not self.joined:
            self.shutdown(False, True)

    def request(self, run, args=()):
        """
        Tuple version of callInThread().
        """
        self.callInThread(run, *args)

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
        if self not in ThreadPool.pools or self.shutting_down:
            return
        self.shutting_down = True
        sdl = len(self.on_shutdowns)
        self.debug("Running shutdown-ers")
        for on_shutdown, ind in zip(self.on_shutdowns, range(sdl)):
            self.debug("%d/%d - %s", ind, sdl, str(on_shutdown))
            on_shutdown()
        self.debug("Joining threads")
        del self.on_shutdowns[:]
        self.joined = True
        threads = copy.copy(self.threads)
        if not execute_remaining:
            self.q._put = types.MethodType(ThreadPool._put, self.q)
        while self.workers:
            self.q.put(threadpool.WorkerStop)
            self.workers -= 1
        for thread in threads:
            if not force:
                thread.join()
            else:
                thread.join(timeout)
                if thread.is_alive():
                    self.warning("Stack trace of probably deadlocked #%d:",
                                 thread.ident)
                    traceback.print_stack(sys._current_frames()[thread.ident])
                    ThreadPool.force_thread_to_stop(thread)
                    self.warning(
                        "Failed to join with thread #%d since the  timeout "
                        "(%.2f sec) was exceeded.%s",
                        thread.ident, timeout, " It was killed."
                        if ThreadPool.thread_can_be_forced_to_stop(thread)
                        else " It was not killed due to the lack of _stop in "
                        " Thread class of the current Python interpreter.")
        ThreadPool.pools.remove(self)
        self.debug("I am destroyed")
        self.shutting_down = False

    @staticmethod
    def thread_can_be_forced_to_stop(thread):
        return hasattr(thread, "_stop") and callable(thread._stop)

    @staticmethod
    def force_thread_to_stop(thread):
        if ThreadPool.thread_can_be_forced_to_stop(thread):
            thread._stop()

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
        sys.exit = ThreadPool.sysexit_initial
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
        for tid, stack in sys._current_frames().items():
            print("-" * 80)
            print("Thread #%d:" % tid)
            traceback.print_stack(stack)

    @staticmethod
    def debug_deadlocks():
        if threading.activeCount() > 1:
            logging.warning("There are currently more than 1 threads still "
                            "running. A deadlock is likely to happen.\n%s",
                            str(threading._active)
                            if hasattr(threading, "_active")
                            else "<unable to list active threads>")
