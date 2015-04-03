# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 4, 2014

Normalizes multichannel byte images according to
dataset mean and dispersion.

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
from veles.memory import Vector
from veles.distributable import IDistributable, TriviallyDistributable
from veles.opencl_types import numpy_dtype_to_opencl
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit


@implementer(IDistributable, IOpenCLUnit, ICUDAUnit)
class MeanDispNormalizer(AcceleratedUnit, TriviallyDistributable):
    """Normalizes multichannel byte images according to
    dataset mean and dispersion.

    Attributes:
        input: minibatch of images (dtype=numpy.uint8,
                                    shape[0]=minibatch_size).
        mean: mean image over the dataset (dtype=numpy.uint8).
        rdisp: 1.0 / dispersion over the dataset (float datatype).
        output: normalized float images of the same dtype as rdisp.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "WORKER")
        super(MeanDispNormalizer, self).__init__(workflow, **kwargs)
        self.input = None
        self.mean = None
        self.rdisp = None
        self.output = Vector()
        self.demand("input", "mean", "rdisp")
        self.global_size = None
        self.local_size = None

    def init_unpickled(self):
        super(MeanDispNormalizer, self).init_unpickled()
        self.sources_["mean_disp_normalizer"] = {}

    def initialize(self, device, **kwargs):
        super(MeanDispNormalizer, self).initialize(device, **kwargs)

        if not isinstance(self.input, Vector) or self.input.mem is None:
            raise error.BadFormatError("input should be assigned as Vector")
        if not isinstance(self.mean, Vector) or self.mean.mem is None:
            raise error.BadFormatError("mean should be assigned as Vector")
        if not isinstance(self.rdisp, Vector) or self.rdisp.mem is None:
            raise error.BadFormatError("rdisp should be assigned as Vector")
        if len(self.input.shape) < 2:
            raise error.BadFormatError("input should be at least 2D")
        sample_size = self.mean.size
        if (self.input.sample_size != sample_size or
                self.rdisp.size != sample_size):
            raise error.BadFormatError("Sample size of input differs from "
                                       "mean-rdisp size")

        if not self.output.mem:
            self.output.reset(numpy.zeros(self.input.shape, self.rdisp.dtype))
        else:
            assert self.output.shape == self.input.shape

        self.init_vectors(self.input, self.mean, self.rdisp, self.output)

    def _gpu_init(self):
        dtype = self.rdisp.dtype
        sample_size = self.mean.size

        defines = {
            "input_type": numpy_dtype_to_opencl(self.input.dtype),
            "mean_type": numpy_dtype_to_opencl(self.mean.dtype),
            "SAMPLE_SIZE": sample_size
        }
        self.build_program(defines, self.__class__.__name__, dtype=dtype)
        self.assign_kernel("normalize_mean_disp")
        self.set_args(self.input, self.mean, self.rdisp, self.output)

    def ocl_init(self):
        self._gpu_init()
        self.global_size = [self.mean.size, self.input.shape[0]]

    def cuda_init(self):
        self._gpu_init()
        self.local_size = 1, 1, 1
        self.global_size = self.mean.size, self.input.shape[0], 1

    def _gpu_run(self):
        self.unmap_vectors(self.input, self.mean, self.rdisp, self.output)
        self.execute_kernel(self.global_size, self.local_size)

    def ocl_run(self):
        self._gpu_run()

    def cuda_run(self):
        self._gpu_run()

    def cpu_run(self):
        self.input.map_read()
        self.mean.map_read()
        self.rdisp.map_read()
        self.output.map_invalidate()

        dtype = self.output.dtype
        self.output.matrix[:] = (
            self.input.matrix.astype(dtype)[:] -
            self.mean.plain.astype(dtype)) * self.rdisp.plain
