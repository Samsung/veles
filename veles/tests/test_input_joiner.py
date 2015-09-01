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


import numpy

from veles.dummy import DummyUnit
from veles.memory import Array
import veles.input_joiner as input_joiner
from veles.tests import AcceleratedTest, assign_backend


class TestInputJoiner(AcceleratedTest):
    ABSTRACT = True

    def test_link_inputs(self):
        self.info("Will test InputJoiner::link_inputs()")
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        u_ab = DummyUnit(a=a, b=b)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        u_c = DummyUnit(c=c)

        obj = input_joiner.InputJoiner(self.parent)
        obj.link_inputs(u_ab, "a", "b")
        obj.link_inputs(u_c, "c")
        tmp = c.mem
        c.mem = None
        self.assertTrue(bool(obj.initialize(device=self.device)))
        c.mem = tmp
        self.assertFalse(bool(obj.initialize(device=self.device)))
        self.assertEqual(obj.num_inputs, 3)
        self.assertEqual(obj.offset_0, 0)
        self.assertEqual(obj.offset_1, 25)
        self.assertEqual(obj.offset_2, 30)
        self.assertEqual(obj.length_0, 25)
        self.assertEqual(obj.length_1, 5)
        self.assertEqual(obj.length_2, 35)

        # Replace one element without copying back to device
        b.map_write()
        b.mem[b.shape[0] // 2] = -1

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

    def test1(self):
        self.info("Will test InputJoiner()")
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=self.device)

        # Replace one element without copying back to device
        b.map_write()
        b.mem[b.shape[0] // 2] = -1

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

    def test2(self):
        self.info("Will test InputJoiner() "
                  "with output size greater than inputs.")
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=self.device)
        a.initialize(self.device)
        b.initialize(self.device)
        c.initialize(self.device)
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

    def test3(self):
        self.info("Will test InputJoiner() "
                  "with output size less than inputs.")
        a = Array()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = Array()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = Array()
        c.mem = numpy.arange(350, dtype=numpy.float32).reshape(10, 35)
        obj = input_joiner.InputJoiner(self.parent, inputs=[a, b, c])
        obj.initialize(device=self.device)
        a.initialize(self.device)
        b.initialize(self.device)
        c.initialize(self.device)
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


@assign_backend("ocl")
class OCLTestInputJoiner(TestInputJoiner):
    pass


@assign_backend("cuda")
class CUDATestInputJoiner(TestInputJoiner):
    pass


@assign_backend("numpy")
class NUMPYTestInputJoiner(TestInputJoiner):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
