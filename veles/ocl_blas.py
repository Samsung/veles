# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 13, 2015

BLAS class for using with ocl backend.

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

from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit
from veles.dummy import DummyWorkflow
from veles.numpy_ext import roundup


@implementer(IOpenCLUnit)
class Builder(AcceleratedUnit):
    def __init__(self, workflow, **kwargs):
        super(Builder, self).__init__(workflow, **kwargs)
        self.source = kwargs["source"]
        self.defines = kwargs["defines"]
        self.kernel_name = kwargs["kernel_name"]
        self.cache_file_name = kwargs["cache_file_name"]
        self.dtype = kwargs["dtype"]

    @property
    def kernel(self):
        return self._kernel_

    def ocl_init(self):
        self.sources_[self.source] = {}
        self.build_program(self.defines, self.cache_file_name, self.dtype)
        self.assign_kernel(self.kernel_name)

    def ocl_run(self):
        pass


class BLAS(object):
    OP_N = 0
    OP_T = 1

    def __init__(self, device):
        self.device = device
        self.kernels = {}

    def veles_gemm(self, transA, transB,
                   rowsCountA, columnCountB, commonSideLength,
                   alpha, A, B, beta, C):
        """Does a matrix multiplication like in CUBLAS.

        Matricies are assumed to be tightly packed and stored like in CUBLAS.
        """
        dtype = alpha.dtype
        key = (transA, transB, rowsCountA, columnCountB, commonSideLength,
               dtype)
        krn_info = self.kernels.get(key)
        if krn_info is None:
            block_size = self.device.device_info.get_block_size(
                kernel="matrix_multiplication", dtype=dtype)
            defines = {
                "BLOCK_SIZE": block_size,
                "B_WIDTH": rowsCountA,
                "A_WIDTH": columnCountB,
                "AB_COMMON": commonSideLength
            }
            if transA == BLAS.OP_T:
                defines["B_COL"] = 1
            else:
                assert transA == BLAS.OP_N
            if transB == BLAS.OP_N:
                defines["A_COL"] = 1
            else:
                assert transB == BLAS.OP_T
            global_size = (roundup(rowsCountA, block_size),
                           roundup(columnCountB, block_size))
            local_size = (block_size, block_size)
            w = DummyWorkflow()
            builder = Builder(
                w, source="gemm", defines=defines, kernel_name="gemm",
                cache_file_name=(
                    "veles_gemm_%s" % "_".join(str(x) for x in key)),
                dtype=dtype)
            builder.initialize(self.device)
            krn_info = (builder.kernel, global_size, local_size)
            self.kernels[key] = krn_info
            del builder
            del w

        # Set the constants and execute the kernel
        krn = krn_info[0]
        # Our kernel stores output in row-major order, so swap A and B
        krn.set_args(B, A, C, alpha, beta)
        global_size = krn_info[1]
        local_size = krn_info[2]
        self.device.queue_.execute_kernel(krn, global_size, local_size,
                                          need_event=False)
