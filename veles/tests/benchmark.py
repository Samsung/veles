"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import logging
import unittest

import veles.opencl as opencl


class Test(unittest.TestCase):
    def add_ref(self, unit):
        pass

    def testBenchmark(self):
        logging.basicConfig(level=logging.DEBUG)
        self.bench = opencl.OpenCLBenchmark(self)
        logging.info("Result: %d points", self.bench.estimate())


if __name__ == "__main__":
    unittest.main()
