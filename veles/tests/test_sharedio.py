"""
Created on Jul 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from multiprocessing import Process
import unittest

from veles.external.txzmq import SharedIO
from veles.pickle2 import pickle


class TestSharedIO(unittest.TestCase):
    DATA = b"Hello, world!"

    def otherReadWrite(self, name, size):
        shmem = SharedIO(name, size)
        shmem.write(TestSharedIO.DATA)

    def testReadWrite(self):
        shmem = SharedIO("test veles", 1024)
        other = Process(target=self.otherReadWrite,
                        args=(shmem.name, shmem.size))
        other.start()
        other.join()
        data = shmem.read(len(TestSharedIO.DATA))
        self.assertEqual(TestSharedIO.DATA, data)

    def testOverflow(self):
        shmem = SharedIO("test veles", 4)
        self.assertRaises(ValueError, shmem.write, TestSharedIO.DATA)

    def testPickleUnpickle(self):
        shmem = SharedIO("test veles", 1024)
        shmem.seek(100)
        ser = pickle.dumps(shmem)
        other = pickle.loads(ser)
        self.assertEqual(shmem.name, other.name)
        self.assertEqual(shmem.size, other.size)
        self.assertEqual(shmem.tell(), other.tell())
        other2 = pickle.loads(ser)
        self.assertEqual(shmem.name, other2.name)
        self.assertEqual(shmem.size, other2.size)
        self.assertEqual(shmem.tell(), other2.tell())
        self.assertEqual(other.shmem, other2.shmem)
        self.assertEqual(other.refs, 2)
        del other2
        self.assertEqual(other.refs, 1)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReadWrite']
    unittest.main()
