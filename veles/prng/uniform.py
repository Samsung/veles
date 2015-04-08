# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 27, 2014

Uniform random generator unit.

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
from zope.interface import implementer

import veles.error as error
from veles.memory import Vector, roundup
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit, \
    INumpyUnit
from veles.prng.random_generator import get


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class Uniform(AcceleratedUnit):
    """Generates random numbers from uniform distribution.

    Attributes:
        num_states: number of random states for parallel generation.
        states: Vector of random states.
        prng: veles.prng.RandomGenerator for initial states generation.
        output_bytes: number of output bytes to generate.
    """

    backend_methods = AcceleratedUnit.backend_methods + ("fill",)

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
        self.sources_["random"] = {}

    def initialize(self, device, **kwargs):
        super(Uniform, self).initialize(device, **kwargs)

        if not self.states or self.states.size != self.num_states * 16:
            self.states.reset(numpy.empty(self.num_states * 16 * 2,
                                          dtype=numpy.uint32))
            self.states.mem[:] = self.prng.randint(0, (1 << 32) + 1,
                                                   self.states.size)

        if not self.output or self.output.nbytes < self.output_bytes:
            self.output_bytes = roundup(self.output_bytes,
                                        self.num_states * 16 * 8)
            self.output.reset(numpy.zeros(self.output_bytes, numpy.uint8))
        else:
            self.output_bytes = self.output.nbytes

        self.init_vectors(self.states, self.output)

    def _gpu_init(self):
        self.build_program({}, "uniform_%d" % self.num_states)

        self.assign_kernel("random_xorshift1024star")
        self.set_args(self.states, self.cl_const, self.output)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = [self.num_states]
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        n = self.num_states
        l = 1
        while not (n & 1) and l < 32:
            n >>= 1
            l <<= 1
        self._global_size = (n, 1, 1)
        self._local_size = (l, 1, 1)

    def _gpu_fill(self, nbytes):
        bytes_per_round = self.num_states * 16 * 8
        nbytes = roundup(nbytes, bytes_per_round)
        if nbytes > self.output.nbytes:
            raise error.Bug("nbytes > self.output.nbytes")
        self.unmap_vectors(self.states, self.output)
        self.cl_const[0] = nbytes // bytes_per_round
        self.set_arg(1, self.cl_const)
        self.execute_kernel(self._global_size, self._local_size)

    def ocl_fill(self, nbytes):
        self._gpu_fill(nbytes)

    def cuda_fill(self, nbytes):
        self._gpu_fill(nbytes)

    def numpy_fill(self, nbytes):
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
        self._backend_fill_(nbytes)

    def ocl_run(self):
        self.ocl_fill(self.output.nbytes)

    def cuda_run(self):
        self.cuda_fill(self.output.nbytes)

    def numpy_run(self):
        self.numpy_fill(self.output.nbytes)
