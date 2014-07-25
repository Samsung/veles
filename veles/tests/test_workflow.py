"""
Created on Jun 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import unittest
from zope.interface.verify import verifyObject

from veles import Workflow
from veles.distributable import IDistributable
from veles.units import TrivialUnit
from veles.tests import DummyLauncher
from veles.workflow import StartPoint
from veles.tests import DummyWorkflow
from veles.pickle2 import pickle


class Test(unittest.TestCase):
    def add_units(self, wf):
        u1 = TrivialUnit(wf, name="unit1")
        u1.tag = 0
        u1.link_from(wf.start_point)
        u2 = TrivialUnit(wf, name="unit1")
        u2.tag = 1
        u2.link_from(u1)
        u3 = TrivialUnit(wf, name="unit1")
        u3.tag = 2
        u3.link_from(u2)
        u4 = TrivialUnit(wf, name="unit2")
        u4.link_from(u3)
        u5 = TrivialUnit(wf, name="aaa")
        u5.link_from(u4)
        wf.end_point.link_from(u5)

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

    def testUnits(self):
        wf = Workflow(DummyLauncher())
        self.add_units(wf)
        units = wf.units
        self.assertTrue(isinstance(units, list))
        self.assertEqual(7, len(units))
        self.assertEqual("Start", units[0].name)
        self.assertEqual("End", units[1].name)
        self.assertEqual("unit1", units[2].name)
        self.assertEqual("unit1", units[3].name)
        self.assertEqual("unit1", units[4].name)
        self.assertEqual("unit2", units[5].name)
        self.assertEqual("aaa", units[6].name)
        units = wf.units_in_dependency_order
        self.assertTrue(isinstance(units, list))
        self.assertEqual(7, len(units))
        self.assertEqual("Start", units[0].name)
        self.assertEqual("unit1", units[1].name)
        self.assertEqual("unit1", units[2].name)
        self.assertEqual("unit1", units[3].name)
        self.assertEqual("unit2", units[4].name)
        self.assertEqual("aaa", units[5].name)
        self.assertEqual("End", units[6].name)

    def testGraph(self):
        wf = Workflow(DummyLauncher())
        self.add_units(wf)
        dot, _ = wf.generate_graph(write_on_disk=False)
        ids = []
        for unit in wf:
            ids.append(hex(id(unit)))
            ids.append(ids[-1])
            ids.append(ids[-1])
        # Move EndPoint to the tail
        backup = ids[3:6]
        ids[3:-3] = ids[6:]
        ids[-3:] = backup
        ids = ids[1:-1]
        valid = ('digraph Workflow {\n'
                 'bgcolor=transparent;\n'
                 '"%s" [fillcolor=lightgrey, gradientangle=90, '
                 'label=<<b>Start</b><br/><font point-size="8">'
                 'veles/workflow.py</font>>, shape=rect, '
                 'style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b>unit1</b><br/><font point-size="8">veles/units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b>unit1</b><br/><font point-size="8">veles/units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b>unit1</b><br/><font point-size="8">veles/units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b>unit2</b><br/><font point-size="8">veles/units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b>aaa</b><br/><font point-size="8">veles/units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s";\n'
                 '"%s" [fillcolor=lightgrey, gradientangle=90, '
                 'label=<<b>End</b><br/><font point-size="8">veles/workflow.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '}') % tuple(ids)
        self.maxDiff = None
        self.assertEqual(valid, dot)

    def testStartPoint(self):
        sp = StartPoint(DummyWorkflow())
        verifyObject(IDistributable, sp)
        sp = pickle.loads(pickle.dumps(sp))
        verifyObject(IDistributable, sp)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testItarator']
    unittest.main()
