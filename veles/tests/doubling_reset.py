#!/usr/bin/python3
"""
Created on February 5, 2015

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
from veles.memory import Vector, assert_addr


def patch(self, instance, shape_func, dtype_func):
    def doubling_reset(mem=None):
        Vector.reset(instance, mem)
        if mem is None:
            return
        instance_name = None
        for k, v in self.__dict__.items():
            if v is instance:
                instance_name = k
                break
        self.debug("Unit test mode: allocating 2x memory for %s",
                   instance_name)
        shape = list(shape_func())
        shape[0] <<= 1
        instance.mem = numpy.zeros(shape, dtype_func())
        instance.initialize(self.device)
        instance.map_write()
        instance.unit_test_mem = instance.mem
        shape[0] >>= 1
        instance.mem = instance.unit_test_mem[:shape[0]]
        assert_addr(instance.mem, instance.unit_test_mem)
        instance.unit_test_mem[shape[0]:] = numpy.nan

    instance.reset = doubling_reset
