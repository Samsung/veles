"""
Created on Apr 4, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import unittest

from veles.calculator import Calculator
from veles.tests.dummy_workflow import DummyWorkflow


class UnitMock(object):
    pass


class CalculatorTester(Calculator):
    def calculate(self):
        self.c = self.a + self.b
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
        calc.a = (u1, "A")
        calc.b = (u2, "B")
        calc.c = (u3, "C")
        calc.run()
        self.assertEqual(16, u1.A)
        self.assertEqual(-4, u2.B)
        self.assertEqual(52, u3.C)


if __name__ == "__main__":
    unittest.main()
