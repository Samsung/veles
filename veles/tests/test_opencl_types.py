#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on April 3, 2014

Unit test for opencl types

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

import logging
import numpy
import unittest

import veles.opencl_types as ot


class TestOpenclTypes(unittest.TestCase):
    def test_numpy_dtype_to_opencl(self):
        logging.info("Will test opencl types")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.float32), "float")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.float64), "double")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.complex64), "float2")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.complex128), "double2")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.int8), "char")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.int16), "short")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.int32), "int")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.int64), "long")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.uint8), "uchar")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.uint16),
                         "ushort")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.uint32), "uint")
        self.assertEqual(ot.numpy_dtype_to_opencl(numpy.uint64),
                         "ulong")
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Will test config unit")
    unittest.main()
