"""
Created on Nov 8, 2013

Will test correctness of OpenCL random xor-shift generator.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
from veles.opencl_units import OpenCLUnit, IOpenCLUnit
import veles.prng as rnd
from veles.tests.dummy_workflow import DummyWorkflow


@implementer(IOpenCLUnit)
class TrivialOpenCLUnit(OpenCLUnit):
    def cpu_run(self):
        pass

    def ocl_run(self):
        pass


class TestRandom1024(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()
        self.chunk = 4
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

        obj = TrivialOpenCLUnit(DummyWorkflow())
        obj.initialize(device=self.device)
        obj.cl_sources_["random.cl"] = {'LOG_CHUNK': self.chunk}
        obj.build_program({}, os.path.join(root.common.cache_dir,
                                           "test_random.cl"))

        krn = obj.get_kernel("random_xorshift1024star")
        krn.set_args(states.devmem, n_rounds, output.devmem)

        self.device.queue_.execute_kernel(krn, (states.mem.shape[0],),
                                          None, need_event=False)

        output.map_read()
        logging.debug("gpu output:")
        logging.debug(output.mem)

        return output.mem

    def _cpu(self):
        states = self.states.copy()
        output = numpy.zeros(self.n_states * 16 * self.n_rounds,
                             dtype=numpy.uint64)
        for i in range(self.n_states):
            offs = i
            s = states[i]
            self.p = 0
            for _ in range(self.n_rounds):
                for _ in range(16):
                    output[offs] = self._next_rand(s)
                    offs += self.n_states
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


class TestRandom128(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()
        self.chunk = 4

    def tearDown(self):
        del self.device

    def _gpu(self):
        states = formats.Vector()
        output = formats.Vector()
        states.mem = self.states.copy()
        output.mem = numpy.zeros(states.mem.size // 2, dtype=numpy.uint64)
        states.initialize(self.device)
        output.initialize(self.device)

        obj = TrivialOpenCLUnit(DummyWorkflow())
        obj.initialize(device=self.device)
        obj.cl_sources_["random.cl"] = {'LOG_CHUNK': self.chunk}
        obj.build_program({}, os.path.join(root.common.cache_dir,
                                           "test_random.cl"))
        obj.assign_kernel("random_xorshift128plus")
        obj._set_args(states, output)
        obj.execute_kernel((output.mem.size >> self.chunk,), None,
                           need_event=False)

        output.map_read()
        logging.debug("opencl output:")
        logging.debug(output.mem)
        return output.mem

    def _cpu(self):
        numpy.seterr(over='ignore')
        states = self.states.copy()
        output = numpy.zeros(states.size // 2, dtype=numpy.uint64)
        for i in range(0, states.size, 2):
            output[i // 2] = rnd.xorshift128plus(states, i)
        numpy.seterr(over='warn')
        logging.debug("cpu output:")
        logging.debug(output)
        return output

    def test(self):
        # 4 here is not related to self.chunk
        self.states = rnd.get().randint(
            0, numpy.iinfo(numpy.uint32).max, 1024 * 4) \
            .astype(numpy.uint32).view(numpy.uint64)
        v_gpu = self._gpu()
        v_cpu = self._cpu()
        self.assertEqual(numpy.count_nonzero(v_gpu - v_cpu), 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
