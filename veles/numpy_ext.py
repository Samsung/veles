# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on March 11, 2015

Numpy extension functions which ensure virtual address persistence.

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


def max_type(num):
    """Returns array converted to supported type with maximum precision.
    """
    return num.astype(numpy.float64)


def eq_addr(a, b):
    return a.__array_interface__["data"][0] == b.__array_interface__["data"][0]


def assert_addr(a, b):
    """Raises an exception if addresses of the supplied arrays differ.
    """
    if not eq_addr(a, b):
        raise ValueError("Addresses of the arrays are not equal.")


def ravel(a):
    """numpy.ravel() with address check.
    """
    b = a.ravel()
    assert_addr(a, b)
    return b


def reshape(a, shape):
    """numpy.reshape() with address check.
    """
    b = a.reshape(shape)
    assert_addr(a, b)
    return b


def reshape_transposed(w):
    """Reshapes weights as if they were transposed.
    """
    a = w.reshape(*w.shape[1::-1])
    assert_addr(a, w)
    return a


def transpose(a):
    """numpy.transpose() with address check.
    """
    b = a.transpose()
    assert_addr(a, b)
    return b


def interleave(arr):
    """Returns the interleaved array.

    Example:
        [10000, 3, 32, 32] => [10000, 32, 32, 3].
    """
    last = arr.shape[-3]
    b = numpy.empty((arr.shape[0],) + arr.shape[2:] + (last,), arr.dtype)
    if len(b.shape) == 4:
        for i in range(last):
            b[:, :, :, i] = arr[:, i, :, :]
    elif len(b.shape) == 3:
        for i in range(last):
            b[:, :, i] = arr[i, :, :]
    else:
        raise ValueError("a should be of shape 4 or 3.")
    return b


def roundup(num, align):
    d = num % align
    if d == 0:
        return num
    return num + (align - d)


class NumDiff(object):
    """Numeric differentiation helper.

    WARNING: it is invalid for single precision float data type.
    """

    def __init__(self):
        self.h = 1.0e-8
        self.points = (2.0 * self.h, self.h, -self.h, -2.0 * self.h)
        self.coeffs = numpy.array([-1.0, 8.0, -8.0, 1.0],
                                  dtype=numpy.float64)
        self.divizor = 12.0 * self.h
        self.errs = numpy.zeros_like(self.points)

    @property
    def derivative(self):
        return (self.errs * self.coeffs).sum() / self.divizor
