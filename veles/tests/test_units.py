"""
Created on Apr 4, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import unittest

from veles.units import Unit
from veles.tests.dummy_workflow import DummyWorkflow


class UnitMock(object):
    pass


class CalculatorTester(Unit):
    def __init__(self, workflow):
        super(CalculatorTester, self).__init__(workflow)
        self.a = None
        self.b = None
        self.c = None

    def run1(self):
        self.c = self.a + self.b

    def run2(self):
        self.a = self.b ** 2


class Test(unittest.TestCase):

    def testCalculate(self):
        w = DummyWorkflow()
        calc = CalculatorTester(w)
        u1 = UnitMock()
        u1.A = 56
        u2 = UnitMock()
        u2.B = -4
        u3 = UnitMock()
        u3.C = 1
        calc.link_attrs(u1, ("a", "A"))
        calc.link_attrs(u2, ("b", "B"))
        calc.link_attrs(u3, ("c", "C"), two_way=True)
        calc.run1()
        self.assertRaises(RuntimeError, calc.run2)
        self.assertEqual(56, u1.A)
        self.assertEqual(-4, u2.B)
        self.assertEqual(52, u3.C)
        self.assertEqual(56, calc.a)
        self.assertEqual(-4, calc.b)
        self.assertEqual(52, calc.c)


if __name__ == "__main__":
    unittest.main()
