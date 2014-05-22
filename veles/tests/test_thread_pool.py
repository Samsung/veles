"""
Created on Oct 8, 2013

Unit test for ThreadPool().

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy.random
import threading
import time
import unittest

import veles.thread_pool as thread_pool


class TestThreadPool(unittest.TestCase):
    def _job(self, n_jobs, data_lock):
        time.sleep(numpy.random.rand() * 2 + 1)
        data_lock.acquire()
        n_jobs[0] -= 1
        data_lock.release()

    def test_32_threads(self):
        logging.info("Will test ThreadPool with 32 max threads.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=32, maxthreads=32,
                                      queue_size=32)
        n = 100
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self._job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")

    def test_320_threads(self):
        logging.info("Will test ThreadPool with 320 max threads.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=32, maxthreads=320,
                                      queue_size=320)
        n = 100
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self._job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")

    def test_0_threads(self):
        logging.info("Will test ThreadPool with minthreads=0.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=0, maxthreads=32,
                                      queue_size=32)
        n = 10
        t0 = time.time()
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self._job, (n_jobs, data_lock))
        t1 = time.time()
        pool.shutdown(execute_remaining=False, force=True, timeout=0)
        t2 = time.time()
        logging.info("Added to queue in %.3f seconds,"
                     " Shutdowned in %.3f seconds." % (t1 - t0, t2 - t1))
        self.assertEqual(
            n_jobs[0], 10, "ThreadPool::shutdown(execute_remaining=False)"
            "is not working as expected.")

    def test_double_shutdown(self):
        logging.info("Will test ThreadPool for double shutdown().")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=1, maxthreads=32,
                                      queue_size=32)
        n = 10
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self._job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")
        pool.shutdown(execute_remaining=True)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.test']
    unittest.main()
