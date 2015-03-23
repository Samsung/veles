"""
Created on Oct 29, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import gc
import numpy

import veles.memory as formats
import veles.input_joiner as input_joiner
from veles.tests import AcceleratedTest, multi_device


class TestInputJoiner(AcceleratedTest):
    def tearDown(self):
        gc.collect()

    def _do_test(self, device):
        a = formats.Vector()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
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
        a = formats.Vector()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
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
        a = formats.Vector()
        a.mem = numpy.arange(250, dtype=numpy.float32).reshape(10, 25)
        b = formats.Vector()
        b.mem = numpy.arange(50, dtype=numpy.float32).reshape(10, 5)
        c = formats.Vector()
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
        self._do_test(None)

    @multi_device()
    def testGPU2(self):
        self.info("Will test InputJoiner() on GPU "
                  "with output size greater than inputs.")
        self._do_tst2(self.device)

    def testCPU2(self):
        self.info("Will test InputJoiner() on CPU "
                  "with output size greater than inputs.")
        self._do_tst2(None)

    @multi_device()
    def testGPU3(self):
        self.info("Will test InputJoiner() on GPU "
                  "with output size less than inputs.")
        self._do_tst3(self.device)

    def testCPU3(self):
        self.info("Will test InputJoiner() on CPU "
                  "with output size less than inputs.")
        self._do_tst3(None)


if __name__ == "__main__":
    AcceleratedTest.main()
