"""
Created on Oct 8, 2013

Unit test for ThreadPool().

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import sys
import threading
import time
import unittest

import veles.thread_pool as thread_pool
from veles.prng import get as get_prng
prng = get_prng()


class TestThreadPool(unittest.TestCase):
    def setUp(self):
        self.sysexit = sys.exit

    def assert_exit(self):
        try:
            self.assertEqual(self.sysexit, sys.exit)
        except AssertionError:
            sys.exit = self.sysexit
            raise

    def _job(self, n_jobs, data_lock):
        with data_lock:
            pass
        time.sleep(prng.rand() + 0.1)
        with data_lock:
            n_jobs[0] -= 1

    def test_pause_resume(self):
        pool = thread_pool.ThreadPool(minthreads=1, maxthreads=1)
        pool.start()

        flag = [False]
        event = threading.Event()

        def set_flag():
            flag[0] = True
            event.set()

        pool.callInThread(set_flag)
        event.wait()
        self.assertTrue(flag[0])
        event.clear()
        flag[0] = False
        pool.pause()
        thread = threading.Thread(target=pool.callInThread, args=(set_flag,))
        thread.start()
        time.sleep(0.01)
        self.assertFalse(flag[0])
        pool.resume()
        thread.join()
        event.wait()
        self.assertTrue(flag[0])
        pool.shutdown()

    def _do(self, threads_min, threads_max):
        logging.info("Will test ThreadPool with %d max threads.", threads_max)
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=threads_min,
                                      maxthreads=threads_max,
                                      queue_size=threads_max)
        pool.start()
        n = 100
        n_jobs = [n]
        for _ in range(n):
            pool.callInThread(self._job, n_jobs, data_lock)
        pool.shutdown(execute_remaining=True)
        self.assertEqual(n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining"
                         "=True) is not working as expected.")
        self.assert_exit()

    def test_32_threads(self):
        self._do(32, 32)

    def test_320_threads(self):
        self._do(32, 320)

    def test_0_threads(self):
        logging.info("Will test ThreadPool with minthreads=0.")
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=0, maxthreads=32,
                                      queue_size=32)
        pool.start()
        pool.silent = True
        n = 10
        n_jobs = [n]
        t0 = time.time()
        with data_lock:
            for _ in range(n):
                pool.callInThread(self._job, n_jobs, data_lock)
        t1 = time.time()
        pool.shutdown(execute_remaining=False, force=True, timeout=0)
        t2 = time.time()
        logging.info("Added to queue in %.3f seconds,"
                     " Shutdowned in %.3f seconds." % (t1 - t0, t2 - t1))
        self.assertEqual(
            n_jobs[0], n, "ThreadPool::shutdown(execute_remaining=False)"
            "is not working as expected.")
        self.assert_exit()

    def test_double_shutdown(self):
        logging.info("Will test ThreadPool for double shutdown().")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(minthreads=1, maxthreads=32,
                                      queue_size=32)
        pool.start()
        n = 10
        for _ in range(n):
            with data_lock:
                n_jobs[0] += 1
            pool.callInThread(self._job, n_jobs, data_lock)
        pool.shutdown(execute_remaining=True, timeout=1.1)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")
        pool.shutdown(execute_remaining=True, timeout=1.1)
        self.assertEqual(
            n_jobs[0], 0, "ThreadPool::shutdown(execute_remaining=True)"
            "is not working as expected.")
        self.assert_exit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.test']
    unittest.main()
