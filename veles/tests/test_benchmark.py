"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.accelerated_units import DeviceBenchmark
from veles.tests import AcceleratedTest, multi_device


class TestBenchmark(AcceleratedTest):
    def setUp(self):
        super(TestBenchmark, self).setUp()
        self.bench = DeviceBenchmark(self.parent)

    @multi_device(True)
    def testBenchmark(self):
        self.bench.initialize(device=self.device)
        self.info("Result: %d points", self.bench.run())


if __name__ == "__main__":
    AcceleratedTest.main()
