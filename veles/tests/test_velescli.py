"""
Created on Jun 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import os
import struct
import tempfile
import unittest

from veles.__main__ import Main
import veles.prng as rnd


class Test(unittest.TestCase):
    def setUp(self):
        self.main = Main()

    def testSeeding(self):
        _, fname = tempfile.mkstemp(prefix="veles-test-seed-")
        with open(fname, 'wb') as fw:
            for i in range(100):
                fw.write(struct.pack('i', i))
        self.main._seed_random(fname + ":100")
        state1 = numpy.random.get_state()
        arr1 = numpy.empty(100)
        rnd.get().fill(arr1)
        self.main._seed_random(fname + ":100")
        state2 = numpy.random.get_state()
        try:
            self.assertTrue((state1[1] == state2[1]).all())
            arr2 = numpy.empty(100)
            rnd.get().fill(arr2)
            self.assertTrue((arr1 == arr2).all())
        except AssertionError:
            os.remove(fname)
            raise


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSeeding']
    unittest.main()
