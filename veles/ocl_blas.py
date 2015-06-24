# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 13, 2015

BLAS class to use with ocl backend.

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

from cuda4py.blas import CUBLAS_OP_N, CUBLAS_OP_T
import numpy
import opencl4py.blas as clblas
import os
import weakref
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit
from veles.config import root
from veles.dummy import DummyWorkflow
from veles.logger import Logger
from veles.numpy_ext import roundup


@implementer(IOpenCLUnit)
class Builder(AcceleratedUnit):
    """Dummy unit for building OpenCL kernels.
    """
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


class OCLBLAS(Logger):
    """Class with BLAS functionality similar to CUBLAS.

    It uses CLBLAS when available or custom kernels otherwise.
    """
    @staticmethod
    def attach_to_device(device):
        if device.blas is None:
            device.blas = OCLBLAS(device)

    def __init__(self, device):
        super(OCLBLAS, self).__init__()
        self._device = weakref.ref(device)
        self.kernels = {}
        self._const_i = numpy.zeros(3, dtype=numpy.uint64)
        try:
            if (root.common.engine.ocl.clBLAS is not True or
                    root.common.precision_level > 0):
                raise ValueError()
            if "CLBLAS_STORAGE_PATH" not in os.environ:
                found = False
                for dirnme in root.common.engine.device_dirs:
                    for path, _, files in os.walk(dirnme):
                        for f in files:
                            if f.endswith(".kdb"):
                                found = True
                                os.environ["CLBLAS_STORAGE_PATH"] = path
                                break
                        if found:
                            break
                    if found:
                        break
            self.blas = clblas.CLBLAS()
            self._sgemm = self.clblas_sgemm
            self._dgemm = self.clblas_dgemm
            self.debug("Using clBLAS for matrix multiplication")
        except (OSError, RuntimeError, ValueError):
            self._sgemm = self.veles_gemm
            self._dgemm = self.veles_gemm
            self.debug("Using Veles OpenCL kernels for matrix multiplication")

    @property
    def device(self):
        return self._device()

    @staticmethod
    def gemm(dtype):
        if dtype == numpy.float32:
            return OCLBLAS.sgemm
        if dtype == numpy.float64:
            return OCLBLAS.dgemm
        raise ValueError("Invalid dtype %s" % dtype)

    def sgemm(self, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C, offsetA=0, offsetB=0, offsetC=0):
        return self._sgemm(
            transA, transB, rowsCountA, columnCountB, commonSideLength,
            alpha, A, B, beta, C,
            offsetA=offsetA, offsetB=offsetB, offsetC=offsetC)

    def dgemm(self, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C, offsetA=0, offsetB=0, offsetC=0):
        return self._dgemm(
            transA, transB, rowsCountA, columnCountB, commonSideLength,
            alpha, A, B, beta, C,
            offsetA=offsetA, offsetB=offsetB, offsetC=offsetC)

    def clblas_sgemm(self, transA, transB,
                     rowsCountA, columnCountB, commonSideLength,
                     alpha, A, B, beta, C, offsetA=0, offsetB=0, offsetC=0):
        """Does a matrix multiplication like in CUBLAS using clBLAS.

        Matricies are assumed to be tightly packed and stored like in CUBLAS.

        Single precision (float) version.
        """
        self.blas.sgemm((self.device.queue_,), clblas.clblasColumnMajor,
                        transA, transB, rowsCountA, columnCountB,
                        commonSideLength, alpha, A, B, beta, C,
                        offsetA=offsetA, offsetB=offsetB, offsetC=offsetC)

    def clblas_dgemm(self, transA, transB,
                     rowsCountA, columnCountB, commonSideLength,
                     alpha, A, B, beta, C, offsetA=0, offsetB=0, offsetC=0):
        """Does a matrix multiplication like in CUBLAS using clBLAS.

        Matricies are assumed to be tightly packed and stored like in CUBLAS.

        Double precision (double) version.
        """
        self.blas.dgemm((self.device.queue_,), clblas.clblasColumnMajor,
                        transA, transB, rowsCountA, columnCountB,
                        commonSideLength, alpha, A, B, beta, C,
                        offsetA=offsetA, offsetB=offsetB, offsetC=offsetC)

    def veles_gemm(self, transA, transB,
                   rowsCountA, columnCountB, commonSideLength,
                   alpha, A, B, beta, C, offsetA=0, offsetB=0, offsetC=0):
        """Does a matrix multiplication like in CUBLAS using custom kernel.

        Matricies are assumed to be tightly packed and stored like in CUBLAS.
        """
        dtype = alpha.dtype
        key = (transA, transB, rowsCountA, columnCountB, commonSideLength,
               dtype)
        krn_info = self.kernels.get(key)
        if krn_info is None:
            block_size, vector_opt = self.device.device_info.get_kernel_bs_vo(
                kernel="matrix_multiplication", dtype=dtype)
            defines = {
                "BLOCK_SIZE": block_size,
                "VECTOR_OPT": int(bool(vector_opt)),
                "B_WIDTH": rowsCountA,
                "A_WIDTH": columnCountB,
                "AB_COMMON": commonSideLength
            }
            if transB == CUBLAS_OP_T:
                defines["A_COL"] = 1
            else:
                assert transB == CUBLAS_OP_N
            if transA == CUBLAS_OP_N:
                defines["B_COL"] = 1
            else:
                assert transA == CUBLAS_OP_T
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
        self._const_i[0:3] = offsetA, offsetB, offsetC
        # Our kernel stores output in row-major order, so swap A and B
        krn.set_args(B, A, C, alpha, beta, self._const_i[1:2],
                     self._const_i[0:1], self._const_i[2:3])
        global_size = krn_info[1]
        local_size = krn_info[2]
        self.device.queue_.execute_kernel(krn, global_size, local_size,
                                          need_event=False)
