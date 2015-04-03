"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 4, 2014

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


import unittest
from zope.interface import implementer

import veles
from veles.config import root
from veles.units import TrivialUnit, Container, IUnit
from veles.pickle2 import pickle
from veles.dummy import DummyWorkflow


class UnitMock(object):
    pass


class CalculatorTester(TrivialUnit):
    def __init__(self, workflow):
        super(CalculatorTester, self).__init__(workflow)
        self.a = None
        self.b = None
        self.c = None

    def run1(self):
        self.c = self.a + self.b

    def run2(self):
        self.a = self.b ** 2


class TestUnit(TrivialUnit):
    def __init__(self, workflow, **kwargs):
        self.was_warning = False
        super(TestUnit, self).__init__(workflow, **kwargs)
        self.a = 1
        self.b = "test"
        self.c = [1, 2, 3]

    def warning(self, *args, **kwargs):
        self.was_warning = True


@implementer(IUnit)
class TestContainer(Container, TrivialUnit):
    pass


class Test(unittest.TestCase):

    def testCalculate(self):
        calc = CalculatorTester(DummyWorkflow())
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

    def testSerialization(self):
        unit = TestUnit(DummyWorkflow())
        unit2 = pickle.loads(pickle.dumps(unit))
        self.assertEqual(unit.name, unit2.name)
        for an in ("a", "b", "c"):
            self.assertEqual(getattr(unit, an), getattr(unit2, an))

    def testValidateKwargs(self):
        bad_kwargs = {"first": root.unit_test,
                      "second": root.nonexistent_shit}
        u = TestUnit(DummyWorkflow(), **bad_kwargs)
        self.assertTrue(u.was_warning)
        u.was_warning = False
        u.initialize(**bad_kwargs)
        self.assertTrue(u.was_warning)

    def testUnitsList(self):
        units = veles.__units__
        self.assertIsInstance(units, set)
        self.assertGreater(len(units), 0)
        print("Number of units:", len(units))

    def testContainerInputs(self):
        c = TestContainer(DummyWorkflow())
        tu = TestUnit(DummyWorkflow())
        tu.demand("attr_name")
        tu.link_attrs(c, "attr_name")
        c.attr_name = 100500
        self.assertEqual(tu.attr_name, 100500)
        tu.initialize()
        outer = TestUnit(DummyWorkflow())
        outer.attr_name = 100
        c.demand("attr_name")
        c.link_attrs(outer, "attr_name")
        c.initialize()
        self.assertEqual(tu.attr_name, 100)
        self.assertEqual(c.attr_name, 100)


if __name__ == "__main__":
    unittest.main()
