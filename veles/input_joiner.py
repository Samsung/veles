# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Oct 29, 2013

Joins several inpus into one continuous output.

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


from __future__ import division
import numpy
from zope.interface import implementer

from veles.memory import Vector
import veles.opencl_types as opencl_types
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit


@implementer(IOpenCLUnit, ICUDAUnit)
class InputJoiner(AcceleratedUnit):
    """Joins several minibatch inputs into one continuous minibatch output.

    Must be assigned before initialize():
        inputs

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        inputs: list of inputs of type formats.Vector().
        output: formats.Vector().
        minibatch_size: size of the minibatch (will be set to the minimum
                        of the first shapes from the inputs
                        if not provided prior to the initialize)
    """
    def __init__(self, workflow, **kwargs):
        super(InputJoiner, self).__init__(workflow, **kwargs)
        self.inputs = kwargs["inputs"]
        self.output = Vector()

    def init_unpickled(self):
        super(InputJoiner, self).init_unpickled()
        self.sources_["join"] = {}

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if not hasattr(value, "__iter__"):
            raise TypeError("inputs must be iterable")
        self._inputs = list(value)
        if len(self._inputs) == 0:
            raise ValueError("inputs may not be empty")

    def initialize(self, device, **kwargs):
        if any(i.mem is None for i in self.inputs):
            # Not yet ready to initialize
            return True

        super(InputJoiner, self).initialize(device=device, **kwargs)

        minibatch_size = min(i.shape[0] for i in self.inputs)
        if any(i.shape[0] > minibatch_size for i in self.inputs):
            self.warning("Detected inputs of different sizes. Sizes will be "
                         "cut to the lowest value (%d)", minibatch_size)

        output_shape = (minibatch_size,
                        sum(i.size // i.shape[0] for i in self.inputs))
        if not self.output:
            self.output.reset(numpy.zeros(output_shape, self.inputs[0].dtype))
        else:
            assert self.output.shape == output_shape

        self.init_vectors(self.output, *self.inputs)

    def _gpu_init(self):
        defines = {
            'etype': opencl_types.numpy_dtype_to_opencl(self.output.dtype),
        }
        self.build_program(
            defines, "%s_%d_%s" %
            (type(self).__name__, self.output.shape[0],
             "_".join(map(str, self.output.shape[1:]))), inputs=self.inputs)
        self.assign_kernel("join")
        self.set_args(self.output, *self.inputs)

    def ocl_init(self):
        self._gpu_init()

    def cuda_init(self):
        self._gpu_init()

    def cpu_run(self):
        self.output.map_invalidate()  # we will update output on CPU
        minibatch_size = self.output.shape[0]
        low = 0
        for inp in self.inputs:
            inp.map_read()
            high = low + inp.size // inp.shape[0]
            if low >= high:
                break
            self.output.mem[:, low:high] = inp[:minibatch_size]
            low = high

    def ocl_run(self):
        self.execute_kernel(*((self.output.shape[0],),) * 2)

    def cuda_run(self):
        self.execute_kernel((1, 1, 1), (self.output.shape[0], 1, 1))
