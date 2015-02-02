"""
Created on Jan 30, 2015

Unit tests for normalization.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import unittest

from veles.normalization import NormalizerRegistry


class LinearCase(unittest.TestCase):
    def test_mean_disp(self):
        nclass = NormalizerRegistry.normalizers["mean_disp"]
        mdn = nclass()
        arr = numpy.random.normal(0, 1, 1000)
        mdn.analyze(arr)
        mdn.normalize(arr)
        # mean should be 0, disp should be near 2 * 3 * 1
        self.assertLess(numpy.max(numpy.abs(arr)), 0.6)
        mdn = nclass()
        arr = numpy.ones((10, 10))
        arr[0, :] = 2
        arr[-1, :] = 0
        mdn.analyze(arr)
        mdn.normalize(arr)
        self.assertEqual(numpy.max(arr), 0.5)
        self.assertEqual(numpy.min(arr), -0.5)

    def test_linear(self):
        pass

    def test_exp(self):
        pass

    def test_pointwise(self):
        pass

    def test_mean_file(self):
        pass

if __name__ == '__main__':
    unittest.main()
