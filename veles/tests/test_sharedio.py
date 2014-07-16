"""
Created on Jul 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from multiprocessing import Process, Lock
import unittest

from veles.external.txzmq import SharedIO


class TestSharedIO(unittest.TestCase):
    DATA = b"Hello, world!"

    def otherReadWrite(self, lock):
        with lock:
            shmem = SharedIO("test veles", 1024)
        shmem.write(TestSharedIO.DATA)

    def testReadWrite(self):
        lock = Lock()
        with lock:
            other = Process(target=self.otherReadWrite, args=(lock,))
            other.start()
            shmem = SharedIO("test veles", 1024)
        other.join()
        data = shmem.read(len(TestSharedIO.DATA))
        self.assertEqual(TestSharedIO.DATA, data)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReadWrite']
    unittest.main()
