#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Unit test for opencl types

Copyright (c) 2013 Samsung Electronics Co., Ltd.
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
