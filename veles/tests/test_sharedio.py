# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 16, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from multiprocessing import Process
import unittest

from veles.txzmq import SharedIO
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
