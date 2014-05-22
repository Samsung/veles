"""
Created on Nov 8, 2013

Will test correctness of OpenCL random xor-shift generator.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
from veles.opencl_units import OpenCLUnit
import veles.random_generator as rnd
from veles.tests.dummy_workflow import DummyWorkflow


class TestRandom(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()
        # rnd.get().seed(numpy.fromfile("%s/veles/znicz/samples/seed" %
        #                                (root.common.veles_dir),
        #                                dtype=numpy.int32, count=1024))

    def tearDown(self):
        del self.device

    def _gpu(self):
        states = formats.Vector()
        output = formats.Vector()

        states.mem = self.states.copy()
        n_rounds = numpy.array([self.n_rounds], dtype=numpy.int32)
        output.mem = numpy.zeros(states.mem.shape[0] * 128 // 8 * n_rounds[0],
                               dtype=numpy.uint64)

        states.initialize(self.device)
        output.initialize(self.device)

        obj = OpenCLUnit(DummyWorkflow())
        obj.initialize(device=self.device)
        obj.cl_sources_["random.cl"] = {}
        obj.build_program({}, os.path.join(root.common.cache_dir,
                                           "test_random.cl"))

        krn = obj.get_kernel("random")
        krn.set_arg(0, states.devmem)
        krn.set_arg(1, output.devmem)
        krn.set_arg(2, n_rounds)

        self.device.queue_.execute_kernel(krn, (states.mem.shape[0],),
                                          None).wait()

        output.map_read()
        logging.debug("gpu output:")
        logging.debug(output.mem)

        return output.mem

    def _cpu(self):
        states = self.states.copy()
        output = numpy.zeros(self.n_states * 16 * self.n_rounds,
                             dtype=numpy.uint64)
        offs = 0
        for i in range(self.n_states):
            s = states[i]
            self.p = 0
            for j in range(self.n_rounds):
                for k in range(16):
                    output[offs] = self._next_rand(s)
                    offs += 1
        logging.debug("cpu output:")
        logging.debug(output)
        return output

    def _next_rand(self, s):
        s0 = numpy.array([s[self.p]], dtype=numpy.uint64)
        self.p = (self.p + 1) & 15
        s1 = numpy.array([s[self.p]], dtype=numpy.uint64)
        s1 ^= s1 << 31
        s1 ^= s1 >> 11
        s0 ^= s0 >> 30
        rr = s0 ^ s1
        s[self.p] = rr[0]
        return (rr * numpy.array([1181783497276652981], dtype=numpy.uint64))[0]

    def test(self):
        self.n_states = 5
        self.n_rounds = 3
        self.states = rnd.get().randint(
            0, 0x100000000, self.n_states * 128 // 4).astype(
            numpy.uint32).view(numpy.uint64).reshape(self.n_states, 16)
        v_gpu = self._gpu()
        v_cpu = self._cpu()
        self.assertEqual(numpy.count_nonzero(v_gpu - v_cpu), 0)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
