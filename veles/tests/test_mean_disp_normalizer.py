"""
Created on Jul 4, 2014

Will test correctness of MeanDispNormalizer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
from veles.formats import Vector
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.random_generator as rnd
from veles.mean_disp_normalizer import MeanDispNormalizer
from veles.tests.dummy_workflow import DummyWorkflow


class TestMeanDispNormalizer(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        dtype = opencl_types.dtypes[root.common.precision_type]
        self.mean = numpy.zeros([256, 256, 4], dtype=dtype)
        rnd.get().fill(self.mean, 0, 255)
        numpy.around(self.mean, 0, self.mean)
        numpy.clip(self.mean, 0, 255)
        self.mean = self.mean.astype(numpy.uint8)
        self.rdisp = numpy.ones(self.mean.shape, dtype=dtype)
        rnd.get().fill(self.rdisp, 0.001, 0.999)
        self.input = numpy.zeros((100,) + self.mean.shape, dtype=dtype)
        rnd.get().fill(self.input, 0, 255)
        numpy.around(self.input, 0, self.input)
        numpy.clip(self.input, 0, 255)
        self.input = self.input.astype(numpy.uint8)

    def test_random(self):
        gpu = self._test_random(self.device)
        cpu = self._test_random(None)
        max_diff = numpy.fabs(cpu - gpu).max()
        self.assertLess(max_diff, 1.0e-5)

    def _test_random(self, device):
        unit = MeanDispNormalizer(DummyWorkflow())
        unit.input = Vector(self.input.copy())
        unit.mean = Vector(self.mean.copy())
        unit.rdisp = Vector(self.rdisp.copy())
        unit.initialize(device)
        unit.run()
        unit.output.map_read()
        self.assertEqual(unit.output.dtype, self.rdisp.dtype)
        if device is not None:
            vv = unit.output.vv[unit.output.shape[0]:]
            nz = numpy.count_nonzero(numpy.isnan(vv))
            self.assertEqual(nz, vv.size, "Overflow occured")
        return unit.output.mem.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
