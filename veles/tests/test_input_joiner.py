# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Oct 29, 2013

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
from veles.backends import NumpyDevice

from veles.memory import Array
import veles.input_joiner as input_joiner
from veles.tests import AcceleratedTest, multi_device


class TestInputJoiner(AcceleratedTest):
    def tearDown(self):
        gc.collect()

    def _do_test(self, device):
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=device)
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(
            numpy.equal(a.mem, obj.output.mem[:, :a.mem.shape[1]]))
        self.assertEqual(nz, a.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(b.mem,
                        obj.output.mem[:, a.mem.shape[1]:a.mem.shape[1] +
                                       b.mem.shape[1]]))
        self.assertEqual(nz, b.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(c.mem,
                        obj.output.mem[:, a.mem.shape[1] + b.mem.shape[1]:]))
        self.assertEqual(nz, c.mem.size, "Failed")

    def _do_tst2(self, device):
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=device)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(
            numpy.equal(a.mem, obj.output.mem[:, :a.mem.shape[1]]))
        self.assertEqual(nz, a.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(b.mem,
                        obj.output.mem[:, a.mem.shape[1]:a.mem.shape[1] +
                                       b.mem.shape[1]]))
        self.assertEqual(nz, b.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(
                c.mem, obj.output.mem[:, a.mem.shape[1] + b.mem.shape[1]:
                                      a.mem.shape[1] +
                                      b.mem.shape[1] +
                                      c.mem.shape[1]]))
        self.assertEqual(nz, c.mem.size, "Failed")
        nz = numpy.count_nonzero(
            obj.output.mem[:, a.mem.shape[1] +
                           b.mem.shape[1] +
                           c.mem.shape[1]:])
        self.assertEqual(nz, 0, "Failed")

    def _do_tst3(self, device):
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=device)
        a.initialize(device)
        b.initialize(device)
        c.initialize(device)
        obj.run()
        obj.output.map_read()
        nz = numpy.count_nonzero(
            numpy.equal(a.mem, obj.output.mem[:, :a.mem.shape[1]]))
        self.assertEqual(nz, a.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(b.mem, obj.output.mem[:,
                                              a.mem.shape[1]:a.mem.shape[1] +
                                              b.mem.shape[1]]))
        self.assertEqual(nz, b.mem.size, "Failed")
        nz = numpy.count_nonzero(
            numpy.equal(c.mem[:, :obj.output.mem.shape[1] -
                              (a.mem.shape[1] + b.mem.shape[1])],
                        obj.output.mem[:, a.mem.shape[1] + b.mem.shape[1]:]))
        self.assertEqual(
            nz, obj.output.mem.shape[0] * (
                obj.output.mem.shape[1] -
                (a.mem.shape[1] + b.mem.shape[1])), "Failed")

    @multi_device()
    def testGPU(self):
        self.info("Will test InputJoiner() on GPU.")
        self._do_test(self.device)

    def testCPU(self):
        self.info("Will test InputJoiner() on CPU.")
        self._do_test(NumpyDevice())

    @multi_device()
    def testGPU2(self):
        self.info("Will test InputJoiner() on GPU "
                  "with output size greater than inputs.")
        self._do_tst2(self.device)

    def testCPU2(self):
        self.info("Will test InputJoiner() on CPU "
                  "with output size greater than inputs.")
        self._do_tst2(NumpyDevice())

    @multi_device()
    def testGPU3(self):
        self.info("Will test InputJoiner() on GPU "
                  "with output size less than inputs.")
        self._do_tst3(self.device)

    def testCPU3(self):
        self.info("Will test InputJoiner() on CPU "
                  "with output size less than inputs.")
        self._do_tst3(NumpyDevice())


if __name__ == "__main__":
    AcceleratedTest.main()
