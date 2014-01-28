"""
Created on Jan 28, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import unittest

import benchmark


class Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.bench = benchmark.OpenCLBenchmark()

    def tearDown(self):
        pass

    def testBenchmark(self):
        logging.info("Result: %d points", self.bench.estimate())


if __name__ == "__main__":
    unittest.main()
