"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import logging
import unittest

import veles.accelerated_units as opencl_units
import veles.backends as opencl
from veles.config import root


root.common.engine.backend = "ocl"


class Test(unittest.TestCase):
    def add_ref(self, unit):
        pass

    def testBenchmark(self):
        self.bench = opencl_units.DeviceBenchmark(self)
        self.bench.initialize(device=opencl.Device())
        logging.info("Result: %d points", self.bench.estimate())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
