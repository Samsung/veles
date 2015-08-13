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

from veles.memory import Array
import veles.opencl_types as opencl_types
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit, \
    INumpyUnit


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class InputJoiner(AcceleratedUnit):
    """Joins several minibatch inputs into one continuous minibatch output.

    Attributes:
        input_0, input_1, ...: inputs of type Array(), created via link_inputs
        offset_0, offset_1, ...: offsets of each input in elements,
                                 have valid values after initialize().
        length_0, length_1, ...: lengths of each input in elements,
                                 have valid values after initialize.
        output: Array()
        minibatch_size: size of the minibatch (will be set to the minimum
                        of the first shapes from the inputs
                        if not provided prior to the initialize)
    """
    def __init__(self, workflow, **kwargs):
        super(InputJoiner, self).__init__(workflow, **kwargs)
        self.output = Array()
        self._num_inputs = 0
        self.inputs = kwargs.get("inputs")

    def init_unpickled(self):
        super(InputJoiner, self).init_unpickled()
        self.sources_["join"] = {}

    @property
    def num_inputs(self):
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, value):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError("num_inputs must be copnvertible to int")
        for x in range(value, self._num_inputs):
            try:
                delattr(self, "input_%d" % x)
                delattr(self, "offset_%d" % x)
                delattr(self, "length_%d" % x)
            except AttributeError:
                pass
        for x in range(self._num_inputs, value):
            setattr(self, "input_%d" % x, None)
            setattr(self, "offset_%d" % x, None)
            setattr(self, "length_%d" % x, None)
        self._num_inputs = value

    @property
    def inputs(self):
        return list(getattr(self, "input_%d" % x)
                    for x in range(self._num_inputs))

    @property
    def offsets(self):
        return list(getattr(self, "offset_%d" % x)
                    for x in range(self._num_inputs))

    @property
    def lengths(self):
        return list(getattr(self, "length_%d" % x)
                    for x in range(self._num_inputs))

    @inputs.setter
    def inputs(self, value):
        if value is None:
            self.num_inputs = 0
            return
        if not hasattr(value, "__iter__"):
            raise TypeError("inputs must be iterable")
        self.num_inputs = len(value)
        for i, inp in enumerate(value):
            setattr(self, "input_%d" % i, inp)

    def link_inputs(self, other, *args):
        """Adds more inputs and links them.

        It will link args to attributes named
        "input_0", "input_1", etc.

        Parameters:
            other: unit from which to link attributes.
            args: attribute names to link.
        """
        if not len(args):
            raise ValueError("args may not be empty")
        num_inputs = self.num_inputs
        self.num_inputs = num_inputs + len(args)
        for arg in args:
            self.link_attrs(other, ("input_%d" % num_inputs, arg))
            num_inputs += 1

    def _init_offset_length_attributes(self):
        """Initializes offset_0, offset_1, ...
                       length_0, length_1, ...
        """
        offset = 0
        for i in range(self.num_inputs):
            inp = getattr(self, "input_%d" % i)
            setattr(self, "offset_%d" % i, offset)
            setattr(self, "length_%d" % i, inp.sample_size)
            offset += inp.sample_size

    def initialize(self, device, **kwargs):
        if any(i.mem is None for i in self.inputs):
            # Not yet ready to initialize
            return True

        self._init_offset_length_attributes()

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

    def numpy_run(self):
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
        for inp in self.inputs:
            inp.unmap()
        self.execute_kernel(*((self.output.shape[0],),) * 2)

    def cuda_run(self):
        for inp in self.inputs:
            inp.unmap()
        # TODO(a.kazantsev): rewrite CUDA kernel for proper grid size
        self.execute_kernel((1, 1, 1), (self.output.shape[0], 1, 1))
