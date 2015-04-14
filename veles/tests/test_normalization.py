# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jan 30, 2015

Unit tests for normalization.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy
import unittest

from veles.normalization import NormalizerRegistry
from veles.prng import get as get_prng
prng = get_prng()


class TestNormalizers(unittest.TestCase):
    def test_mean_disp(self):
        nclass = NormalizerRegistry.normalizers["mean_disp"]
        mdn = nclass()
        arr = numpy.ones((10, 10), dtype=numpy.float32)
        arr[0, :] = 2
        arr[-1, :] = 0
        mdn.analyze(arr)
        mdn.normalize(arr)
        self.assertEqual(numpy.max(arr), 0.5)
        self.assertEqual(numpy.min(arr), -0.5)

    def test_linear(self):
        nclass = NormalizerRegistry.normalizers["linear"]
        ln = nclass()
        arr = numpy.ones((3, 10), dtype=numpy.float32)
        arr[:, 0] = 2
        arr[:, -1] = 0
        ln.analyze(arr)
        ln.normalize(arr)
        self.assertEqual(numpy.max(arr), 1.0)
        self.assertEqual(numpy.min(arr), -1.0)
        ln = nclass(interval=(-2, 2))
        ln.analyze(arr)
        ln.normalize(arr)
        self.assertEqual(numpy.max(arr), 2.0)
        self.assertEqual(numpy.min(arr), -2.0)

    def test_linear_uniform(self):
        nclass = NormalizerRegistry.normalizers["linear"]
        ln = nclass()
        arr = numpy.ones((3, 10), dtype=numpy.float32)
        arr[0, :5] = 0
        ln.analyze(arr)
        ln.normalize(arr)
        self.assertTrue((arr[0] == (-1,) * 5 + (1,) * 5).all())
        self.assertTrue((arr[1] == 0).all())
        self.assertTrue((arr[2] == 0).all())

    def test_linear_complex(self):
        nclass = NormalizerRegistry.normalizers["linear"]
        ln = nclass(interval=(0, 1))
        arr = numpy.ones((3, 10), dtype=numpy.float32)
        arr[0, :5] = 0
        ln.analyze(arr)
        ln.normalize(arr)
        self.assertTrue((arr[0] == (0,) * 5 + (1,) * 5).all())
        self.assertTrue((arr[1] == 0.5).all())
        self.assertTrue((arr[2] == 0.5).all())

    def test_exp(self):
        nclass = NormalizerRegistry.normalizers["exp"]
        ln = nclass()
        arr = numpy.array([prng.normal(0, 1, 10) for _ in range(4)],
                          numpy.float32)
        ln.analyze(arr)
        ln.normalize(arr)
        self.assertGreater(numpy.min(arr), 0)
        self.assertLess(numpy.max(arr), 1)

    def test_pointwise(self):
        nclass = NormalizerRegistry.normalizers["pointwise"]
        pwn = nclass()
        arr = numpy.array([[1, 2, 3], [6, 5, 4], [7, 8, 9], [10, 11, 12]],
                          dtype=numpy.float32)
        backup = arr.copy()
        pwn.analyze(arr)
        pwn.normalize(arr)
        self.assertLess(numpy.sum(numpy.abs(numpy.array(
            [[-1, -1, -1],
             [0.5 / 4.5, -1.5 / 4.5, -3.5 / 4.5],
             [1.5 / 4.5, 1.5 / 4.5, 1.5 / 4.5],
             [1, 1, 1]]) - arr)), 0.001)
        backup *= 2
        pwn.analyze(backup)
        pwn.normalize(backup)
        self.assertLess(numpy.sum(numpy.abs(backup[-1] - arr[-1])), 0.001)
        self.assertGreater(numpy.sum(numpy.abs(backup[0] - arr[0])), 0.5)

    def test_mean_external(self):
        nclass = NormalizerRegistry.normalizers["external_mean"]
        men = nclass(mean_source=numpy.ones((4, 3), dtype=numpy.float64))
        arr = numpy.array([[1, 2, 3], [6, 5, 4], [7, 8, 9], [10, 11, 12]],
                          dtype=numpy.float32)
        men.analyze(arr)
        men.normalize(arr)
        self.assertLess(numpy.sum(numpy.abs(numpy.array(
            [[0, 1, 2], [5, 4, 3], [6, 7, 8], [9, 10, 11]]) - arr)), 0.001)

    def test_mean_internal(self):
        nclass = NormalizerRegistry.normalizers["internal_mean"]
        men = nclass(mean_source=numpy.ones((4, 3), dtype=numpy.float64))
        arr = numpy.array([[1, 2, 3], [6, 5, 4], [7, 8, 9], [10, 11, 12]],
                          dtype=numpy.float32)
        narr = arr - numpy.array([
            (1 + 6 + 7 + 10) / 4.0, (2 + 5 + 8 + 11) / 4.0,
            (3 + 4 + 9 + 12) / 4.0])
        men.analyze(arr)
        men.normalize(arr)
        self.assertLess(numpy.sum(numpy.abs(narr - arr)), 0.001)

    def test_none(self):
        nclass = NormalizerRegistry.normalizers["none"]
        nn = nclass()
        arr = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.float32)
        backup = arr.copy()
        nn.analyze(arr)
        nn.normalize(arr)
        self.assertTrue((arr == backup).all())

if __name__ == '__main__':
    unittest.main()
