"""
Created on Jun 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import unittest

from veles import Workflow
from veles.units import TrivialUnit
from veles.tests import DummyLauncher


class Test(unittest.TestCase):
    def add_units(self, wf):
        u1 = TrivialUnit(wf, name="unit1")
        u1.tag = 0
        u2 = TrivialUnit(wf, name="unit1")
        u2.tag = 1
        u3 = TrivialUnit(wf, name="unit1")
        u3.tag = 2
        TrivialUnit(wf, name="unit2")
        TrivialUnit(wf, name="aaa")

    def testIterator(self):
        wf = Workflow(DummyLauncher())
        self.add_units(wf)
        self.assertEqual(7, len(wf))
        units = list(wf)
        self.assertEqual(7, len(units))
        self.assertEqual("Start", units[0].name)
        self.assertEqual("End", units[1].name)
        self.assertEqual("unit1", units[2].name)
        self.assertEqual("unit1", units[3].name)
        self.assertEqual("unit1", units[4].name)
        self.assertEqual("unit2", units[5].name)
        self.assertEqual("aaa", units[6].name)
        self.assertEqual(0, units[2].tag)
        self.assertEqual(1, units[3].tag)
        self.assertEqual(2, units[4].tag)

    def testIndex(self):
        wf = Workflow(DummyLauncher())
        self.add_units(wf)
        unit1 = wf["unit1"]
        self.assertTrue(isinstance(unit1, list))
        self.assertEqual(3, len(unit1))
        self.assertEqual(0, unit1[0].tag)
        self.assertEqual("unit1", unit1[0].name)
        self.assertEqual(1, unit1[1].tag)
        self.assertEqual("unit1", unit1[1].name)
        self.assertEqual(2, unit1[2].tag)
        self.assertEqual("unit1", unit1[2].name)
        unit2 = wf["unit2"]
        self.assertTrue(isinstance(unit2, TrivialUnit))
        self.assertEqual("unit2", unit2.name)
        raises = False
        try:
            wf["fail"]
        except KeyError:
            raises = True
        self.assertTrue(raises)
        unit = wf[0]
        self.assertEqual("Start", unit.name)
        unit = wf[1]
        self.assertEqual("End", unit.name)
        unit = wf[2]
        self.assertEqual(0, unit.tag)
        self.assertEqual("unit1", unit.name)
        unit = wf[3]
        self.assertEqual(1, unit.tag)
        self.assertEqual("unit1", unit.name)
        unit = wf[4]
        self.assertEqual(2, unit.tag)
        self.assertEqual("unit1", unit.name)
        unit = wf[5]
        self.assertEqual("unit2", unit.name)
        unit = wf[6]
        self.assertEqual("aaa", unit.name)
        raises = False
        try:
            wf[7]
        except IndexError:
            raises = True
        self.assertTrue(raises)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testItarator']
    unittest.main()
