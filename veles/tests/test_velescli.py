"""
Created on Jun 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import os
import struct
import sys
import tempfile
import unittest

from veles.__main__ import Main
import veles.prng as rnd
from veles.workflow import Workflow


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

    def testRun(self):
        argv = sys.argv
        sys.argv = [argv[0], "-s", "-p", "", __file__, __file__]
        self.main.run()
        self.assertTrue(Workflow.run_was_called)


def run(load, main):
    wf, _ = load(Workflow)
    wf.end_point.link_from(wf.start_point)
    main()
    Workflow.run_was_called = True

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSeeding']
    unittest.main()
