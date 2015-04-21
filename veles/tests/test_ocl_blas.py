# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 14, 2015

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

import gc
import numpy

from veles.memory import Vector
from veles.ocl_blas import BLAS
from veles.tests import AcceleratedTest, assign_backend
import veles.prng as prng


class TestOCLBLASBase(AcceleratedTest):
    ABSTRACT = True

    def test_veles_blas(self):
        blas = BLAS(self.device)
        for _ in range(2):
            self._test_random(blas, 17, 1999, 231)
            gc.collect()
            self._test_random(blas, 7, 9, 8)
            gc.collect()
            self._test_random(blas, 9, 7, 800)
            gc.collect()
            self._test_random(blas, 1, 1, 1)
            gc.collect()
            self._test_random(blas, 7777, 17, 219)
            gc.collect()
            self._test_random(blas, 1777, 1999, 2119)
            gc.collect()
        del blas
        gc.collect()

    def _test_random(self, blas, a_size, b_size, common_size):
        rnd = prng.RandomGenerator(None)
        rnd.seed(123)

        a = Vector(numpy.zeros([a_size, common_size], dtype=self.dtype))
        b = Vector(numpy.zeros([b_size, common_size], dtype=self.dtype))
        c = Vector(numpy.zeros([a_size, b_size], dtype=self.dtype))

        rnd.fill(a.mem)
        rnd.fill(b.mem)

        c_gold = numpy.dot(a.mem, b.mem.transpose()).transpose().ravel()

        at = a.mem.reshape(tuple(reversed(a.shape))).transpose()
        bt = b.mem.reshape(tuple(reversed(b.shape))).transpose()
        c_gold_t = numpy.dot(at, bt.transpose()).transpose().ravel()

        a.initialize(self.device)
        b.initialize(self.device)
        c.initialize(self.device)

        alpha = numpy.ones(1, dtype=self.dtype)
        beta = numpy.zeros(1, dtype=self.dtype)

        blas.veles_gemm(BLAS.OP_N, BLAS.OP_T, a_size, b_size, common_size,
                        alpha, a.devmem, b.devmem, beta, c.devmem)
        c.map_read()
        max_diff = numpy.fabs(c.plain - c_gold).max()
        self.info("max_diff = %.6f", max_diff)
        self.assertLess(max_diff, 1.0e-4)

        c.unmap()
        blas.veles_gemm(BLAS.OP_T, BLAS.OP_N, a_size, b_size, common_size,
                        alpha, a.devmem, b.devmem, beta, c.devmem)
        c.map_read()
        max_diff = numpy.fabs(c.plain - c_gold_t).max()
        self.info("max_diff = %.6f", max_diff)
        self.assertLess(max_diff, 1.0e-4)


@assign_backend("ocl")
class OpenCLTestOCLBLAS(TestOCLBLASBase):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
