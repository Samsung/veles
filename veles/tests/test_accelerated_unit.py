"""
Created on Feb 10, 2015

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import os
import unittest
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit
from veles.dummy import DummyWorkflow


@implementer(IOpenCLUnit, ICUDAUnit)
class TestAcceleratedUnit(AcceleratedUnit):
    def cpu_run(self):
        pass

    def ocl_init(self):
        pass

    def ocl_run(self):
        pass

    def cuda_init(self):
        pass

    def cuda_run(self):
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
        unit.sources_["entry"] = {"A": 1, "B": 2}
        unit.sources_["fourth"] = {"A": 3, "B": 4}
        src, _ = unit._generate_source(
            defines, include_dirs, "float", "cc", template_kwargs)
        self.assertEqual(
            """#define GPU_FORCE_64BIT_PTR 1
#define PRECISION_LEVEL 0
#define dtype float
#define mydef 100500
#define sizeof_dtype 4
#define A 1
#define B 2

// #include "entry.jcc"
Some text on top


// #include "first.jcc"

1

2

3



// #include "fourth.jcc"
Some text from fourth.jcc

4
// END OF "fourth.jcc"

// END OF "first.jcc"


// #include "second.jcc"
Contents of second.jcc
======================


11

12

13

// END OF "second.jcc"

#include "third.cc"

Some text on bottom
// END OF "entry.jcc"
#undef A
#undef B

#define A 3
#define B 4

// #include "fourth.jcc"
Some text from fourth.jcc

4
// END OF "fourth.jcc"
#undef A
#undef B
""", src)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
