"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import logging
import unittest

from veles.accelerated_units import DeviceBenchmark
from veles.backends import Device
from veles.config import root


root.common.engine.backend = "ocl"


class Test(unittest.TestCase):
    def add_ref(self, unit):
        pass

    def testBenchmark(self):
        self.bench = DeviceBenchmark(self)
        self.bench.initialize(device=Device())
        logging.info("Result: %d points", self.bench.estimate())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
