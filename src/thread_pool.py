"""
Created on Jan 21, 2014

@author: Kazantsev Alexey <a.kazantsev@samsung.com>,
         Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import copy
import logging
from queue import Queue
import signal
import sys
import types
from twisted.python import threadpool


class ThreadPool(threadpool.ThreadPool):
    """
    Pool of threads.
    """

    sysexit = sys.exit
    sigint = None
    pools = []

    def __init__(self, minthreads=5, maxthreads=20, name=None):
        """
        Creates a new thread pool and starts it.
        """
        super(ThreadPool, self).__init__()
        self.start()
        self.on_shutdowns = []
        if not ThreadPool.pools:
            sys.exit = ThreadPool.exit
            ThreadPool.sigint = signal.signal(signal.SIGINT,
                                              ThreadPool.signal_handler)
        ThreadPool.pools.append(self)

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

    def _put(self, item):
        """
        Private method used by shutdown() to redefine Queue's _put().
        """
        self.queue.appendleft(item)

    def shutdown(self, execute_remaining=False, force=False, timeout=0.25):
        """Safely brings thread pool down.
        """
        for on_shutdown in self.on_shutdowns:
            on_shutdown()
        self.on_shutdowns.clear()
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
                    thread._stop()
                    logging.warning("Failed to join with thread #%d " +
                                    "since the timeout (%.2f sec) was " +
                                    "exceeded. It was killed.",
                                    thread.ident, timeout)
        sys.exit = self.sysexit

    @staticmethod
    def exit(retcode=0):
        """
        Terminates the running program safely.
        """
        for pool in ThreadPool.pools:
            pool.shutdown()
        sys.exit = ThreadPool.sysexit
        sys.exit(retcode)

    @staticmethod
    def signal_handler(signal, frame):
        for pool in ThreadPool.pools:
            pool.shutdown(False, True)
        ThreadPool.sigint(signal, frame)
