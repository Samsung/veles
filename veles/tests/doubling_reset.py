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

Created on February 5, 2015

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
from veles.memory import Array, assert_addr


def patch(self, instance, shape_func, dtype_func):
    def doubling_reset(mem=None):
        Array.reset(instance, mem)
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
