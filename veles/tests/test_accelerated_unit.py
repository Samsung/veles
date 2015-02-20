"""
Created on Feb 10, 2015

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import os
import unittest
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit
from veles.dummy import DummyWorkflow


@implementer(IOpenCLUnit)
class TestAcceleratedUnit(AcceleratedUnit):
    def cpu_run(self):
        pass

    def ocl_init(self):
        pass

    def ocl_run(self):
        pass

    def initialize(self, device, **kwargs):
        pass


class Test(unittest.TestCase):
    def testGenerateSource(self):
        unit = TestAcceleratedUnit(DummyWorkflow())
        defines = {"mydef": "100500"}
        basedir = os.path.dirname(os.path.abspath(__file__))
        include_dirs = [os.path.join(basedir, "res/code")]
        template_kwargs = {"var": [1, 2, 3]}
        unit.sources_["entry"] = {}
        src, _ = unit._generate_source(
            defines, include_dirs, "float", "cc", template_kwargs)
        self.assertEqual(
            """#define sizeof_dtype 4
#define mydef 100500
#define dtype float
#define PRECISION_LEVEL 0
#define GPU_FORCE_64BIT_PTR 0

// #include "entry.jcc"
Some text on top


// #include "first.jcc"

1

2

3



// #include "fourth.jcc"
Some text from fourth.jcc

4


// #include "second.jcc"
Contents of second.jcc
======================


11

12

13

#include "third.cc"

Some text on bottom""", src)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
