"""
Created on Jan 28, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""
import logging
import unittest

import veles.benchmark


class Test(unittest.TestCase):
    def add_ref(self, unit):
        pass

    def testBenchmark(self):
        logging.basicConfig(level=logging.DEBUG)
        self.bench = veles.benchmark.OpenCLBenchmark(self)
        logging.info("Result: %d points", self.bench.estimate())


if __name__ == "__main__":
    unittest.main()
