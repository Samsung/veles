"""
Created on Jul 27, 2014

Uniform random generator unit.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
from zope.interface import implementer

import veles.error as error
from veles.formats import Vector, roundup
from veles.opencl_units import OpenCLUnit, IOpenCLUnit
from veles.prng import get


@implementer(IOpenCLUnit)
class Uniform(OpenCLUnit):
    """Generates random numbers from uniform distribution.

    Attributes:
        num_states: number of random states for parallel generation.
        states: Vector of random states.
        prng: veles.prng.RandomGenerator for initial states generation.
        output_bytes: number of output bytes to generate.
    """
    def __init__(self, workflow, **kwargs):
        super(Uniform, self).__init__(workflow, **kwargs)
        self.num_states = kwargs.get("num_states", 256)
        self.states = Vector()
        self.prng = kwargs.get("prng", get())
        self.output_bytes = kwargs.get("output_bytes", 0)
        self.output = Vector()
        self.cl_const = numpy.zeros(1, dtype=numpy.int32)

    def init_unpickled(self):
        super(Uniform, self).init_unpickled()
        self.cl_sources_["random.cl"] = {}

    def initialize(self, device, **kwargs):
        super(Uniform, self).initialize(device, **kwargs)

        if not self.states or self.states.size != self.num_states:
            self.states.reset()
            self.states.mem = numpy.empty(self.num_states * 16 * 2,
                                          dtype=numpy.uint32)
            self.states.mem[:] = self.prng.randint(0, (1 << 32) + 1,
                                                   self.states.size)

        if not self.output or self.output.nbytes < self.output_bytes:
            self.output.reset()
            self.output_bytes = roundup(self.output_bytes,
                                        self.num_states * 16 * 8)
            self.output.mem = numpy.zeros(self.output_bytes, dtype=numpy.uint8)
        else:
            self.output_bytes = self.output.nbytes

        self.states.initialize(self)
        self.output.initialize(self)

        if self.device is None:
            return

        self.build_program({}, "uniform_%d.cl" % self.num_states)

        self.assign_kernel("random_xorshift1024star")
        self.set_args(self.states, self.cl_const, self.output)

    def fill_ocl(self, nbytes):
        bytes_per_round = self.num_states * 16 * 8
        nbytes = roundup(nbytes, bytes_per_round)
        if nbytes > self.output.nbytes:
            raise error.Bug("nbytes > self.output.nbytes")
        self.states.unmap()
        self.output.unmap()
        self.cl_const[0] = nbytes // bytes_per_round
        self.set_arg(1, self.cl_const)
        self.execute_kernel([self.num_states], None)

    def fill_cpu(self, nbytes):
        bytes_per_round = self.num_states * 16 * 8
        nbytes = roundup(nbytes, bytes_per_round)
        if nbytes > self.output.nbytes:
            raise error.Bug("nbytes > self.output.nbytes")
        self.states.map_write()
        self.output.map_invalidate()
        n_rounds = nbytes // bytes_per_round

        u64 = numpy.array([1181783497276652981], dtype=numpy.uint64)
        s0 = numpy.zeros(1, dtype=numpy.uint64)
        s1 = numpy.zeros(1, dtype=numpy.uint64)

        states = self.states.mem.view(dtype=numpy.uint64)
        states = states.reshape(states.size // 16, 16)
        output = self.output.mem.view(dtype=numpy.uint64)
        for i in range(self.num_states):
            offs = i
            s = states[i]
            self.p = 0
            for _round in range(n_rounds):
                for _iter in range(16):
                    output[offs] = self._next_rand(s, s0, s1, u64)
                    offs += self.num_states

    def _next_rand(self, s, s0, s1, u64):
        s0[0] = s[self.p]
        self.p = (self.p + 1) & 15
        s1[0] = s[self.p]
        s1 ^= s1 << 31
        s1 ^= s1 >> 11
        s0 ^= s0 >> 30
        s0 ^= s1
        s[self.p] = s0[0]
        return (s0 * u64)[0]

    def fill(self, nbytes):
        return (self.fill_ocl(nbytes) if self.device is not None
                else self.fill_cpu(nbytes))

    def ocl_run(self):
        self.fill_ocl(self.output.nbytes)

    def cpu_run(self):
        self.fill_cpu(self.output.nbytes)
