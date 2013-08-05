"""
Created on Jul 12, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import threading
import traceback


class ThreadPool(object):
    """Pool of threads.

    Attributes:
        sem_: semaphore.
        queue: queue of requests.
        total_threads: number of threads in the pool.
        free_threads: number of free threads in the pool.
    """
    def __init__(self, max_free_threads=10):
        self.sem_ = threading.Semaphore(0)
        self.lock_ = threading.Lock()
        self.exit_lock_ = threading.Lock()
        self.queue = []
        self.total_threads = 0
        self.free_threads = 0
        self.max_free_threads = max_free_threads
        self.exit_lock_.acquire()
        threading.Thread(target=self.pool_cleaner).start()
        self.sysexit = sys.exit
        sys.exit = self.exit

    def exit(self, retcode=0):
        self.shutdown()
        self.sysexit(retcode)

    def pool_cleaner(self):
        """Monitors request queue and executes requests,
            launching new threads if neccessary.
        """
        self.lock_.acquire()
        self.total_threads += 1
        self.lock_.release()
        while True:
            self.lock_.acquire()
            self.free_threads += 1
            if self.free_threads > self.max_free_threads:
                self.free_threads -= 1
                self.total_threads -= 1
                self.lock_.release()
                return
            self.lock_.release()
            try:
                self.sem_.acquire()
            except:  # return in case of broken semaphore
                self.lock_.acquire()
                self.free_threads -= 1
                self.total_threads -= 1
                if self.total_threads <= 0:
                    self.exit_lock_.release()
                self.lock_.release()
                return
            self.lock_.acquire()
            self.free_threads -= 1
            if self.free_threads <= 0:
                threading.Thread(target=self.pool_cleaner).start()
            try:
                (run, args) = self.queue.pop(0)
            except:
                self.total_threads -= 1
                self.lock_.release()
                return
            self.lock_.release()
            try:
                run(*args)
            except:
                # TODO(a.kazantsev): add good error handling here.
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)

    def request(self, run, args=()):
        """Adds request for execution to the queue.
        """
        self.lock_.acquire()
        self.queue.append((run, args))
        self.lock_.release()
        if self.sem_ != None:
            self.sem_.release()

    def shutdown(self):
        """Safely shutdowns thread pool.
        """
        sem_ = self.sem_
        self.sem_ = None
        self.lock_.acquire()
        for i in range(0, self.free_threads):
            sem_.release()
        self.lock_.release()
        self.exit_lock_.acquire()


pool = ThreadPool()
