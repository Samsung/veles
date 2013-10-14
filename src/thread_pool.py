"""
Created on Jul 12, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import threading
import queue
import traceback


global_lock = threading.Lock()


class ThreadPool(object):
    """Pool of threads.

    Attributes:
        sem_: semaphore.
        queue: queue of requests.
        total_threads: number of threads in the pool.
        free_threads: number of free threads in the pool.
        max_free_threads: maximum number of free threads in the pool.
        max_threads: maximum number of executing threads in the pool.
        max_enqueued_tasks: maximum number of tasks enqueued for execution,
                            request() will block if that count is reached.
    """
    def __init__(self, max_free_threads=32, max_threads=512,
                 max_enqueued_tasks=2048):
        self.lock_ = threading.Lock()
        self.exit_lock_ = threading.Lock()
        self.queue_ = queue.Queue(max_enqueued_tasks)
        self.total_threads = 0
        self.free_threads = 0
        self.max_free_threads = max_free_threads
        self.max_threads = max_threads
        self.exit_lock_.acquire()
        threading.Thread(target=self.pool_cleaner).start()
        global_lock.acquire()
        self.sysexit = sys.exit
        sys.exit = self.exit
        global_lock.release()

    def exit(self, retcode=0):
        self.shutdown()
        self.sysexit(retcode)

    def pool_cleaner(self):
        """Monitors request queue and executes requests,
            launching new threads if necessary.
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
                (run, args) = self.queue_.get()
                if self.queue_ == None:
                    raise Exception()
            except:  # return in case of broken queue
                return self.broken_queue()
            self.lock_.acquire()
            self.free_threads -= 1
            if (self.free_threads <= 0 and
                self.total_threads < self.max_threads):
                threading.Thread(target=self.pool_cleaner).start()
            self.lock_.release()
            try:
                run(*args)
            except:
                # TODO(a.kazantsev): add good error handling here.
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
            try:
                self.queue_.task_done()
            except:
                return self.broken_queue()

    def broken_queue(self):
        self.lock_.acquire()
        self.free_threads -= 1
        self.total_threads -= 1
        if self.total_threads <= 0:
            self.exit_lock_.release()
        self.lock_.release()

    def request(self, run, args=()):
        """Adds request for execution to the queue.
        """
        self.queue_.put((run, args))

    def shutdown(self, execute_remaining=False):
        """Safely shutdowns thread pool.
        """
        self.lock_.acquire()
        if self.queue_ == None:
            self.lock_.release()
            return
        self.lock_.release()
        if execute_remaining:
            self.queue_.join()
        queue_ = self.queue_
        self.queue_ = None
        self.lock_.acquire()
        for i in range(0, self.free_threads):
            queue_.put((None, None))
        if self.total_threads <= 0:
            self.lock_.release()
            return
        self.lock_.release()
        self.exit_lock_.acquire()
        global_lock.acquire()
        sys.exit = self.sysexit
        global_lock.release()
