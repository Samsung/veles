# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jun 16, 2014

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


import gc
import six
import unittest
import weakref
from zope.interface.verify import verifyObject
from veles.snapshotter import SnapshotterBase

from veles.workflow import Workflow
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
        self.assertEqual("Start of Workflow", units[0].name)
        self.assertEqual("End of Workflow", units[1].name)
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
        self.assertEqual("Start of Workflow", unit.name)
        unit = wf[1]
        self.assertEqual("End of Workflow", unit.name)
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
        self.assertEqual("Start of Workflow", units[0].name)
        self.assertEqual("End of Workflow", units[1].name)
        self.assertEqual("unit1", units[2].name)
        self.assertEqual("unit1", units[3].name)
        self.assertEqual("unit1", units[4].name)
        self.assertEqual("unit2", units[5].name)
        self.assertEqual("aaa", units[6].name)
        units = wf.units_in_dependency_order
        self.assertTrue(hasattr(units, "__iter__"))
        units = list(units)
        self.assertEqual(7, len(units))
        self.assertEqual("Start of Workflow", units[0].name)
        self.assertEqual("unit1", units[1].name)
        self.assertEqual("unit1", units[2].name)
        self.assertEqual("unit1", units[3].name)
        self.assertEqual("unit2", units[4].name)
        self.assertEqual("aaa", units[5].name)
        self.assertEqual("End of Workflow", units[6].name)

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
                 'mindist=0.5;\n'
                 'outputorder=edgesfirst;\n'
                 'overlap=false;\n'
                 '"%s" [fillcolor=lightgrey, gradientangle=90, '
                 'label=<<b><font point-size="18">Start of Workflow</font>'
                 '</b><br/><font point-size="14">'
                 'plumbing.py</font>>, shape=rect, '
                 'style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b><font point-size="18">unit1</font></b><br/>'
                 '<font point-size="14">units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b><font point-size="18">unit1</font></b><br/>'
                 '<font point-size="14">units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b><font point-size="18">unit1</font></b><br/>'
                 '<font point-size="14">units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b><font point-size="18">unit2</font></b><br/>'
                 '<font point-size="14">units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=white, gradientangle=90, '
                 'label=<<b><font point-size="18">aaa</font></b><br/>'
                 '<font point-size="14">units.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '"%s" -> "%s"  [penwidth=3, weight=100];\n'
                 '"%s" [fillcolor=lightgrey, gradientangle=90, '
                 'label=<<b><font point-size="18">End of Workflow</font>'
                 '</b><br/><font point-size="14">plumbing.py'
                 '</font>>, shape=rect, style="rounded,filled"];\n'
                 '}') % tuple(ids)
        self.maxDiff = None
        self.assertEqual(valid, dot)

    def testStartPoint(self):
        dwf = DummyWorkflow()
        sp = StartPoint(dwf)
        verifyObject(IDistributable, sp)
        sp = pickle.loads(pickle.dumps(sp))
        verifyObject(IDistributable, sp)
        self.assertEqual(sp.workflow, None)
        del dwf

    if six.PY3:
        def testWithDestruction(self):
            flag = [False, False]

            class MyUnit(TrivialUnit):
                def __del__(self):
                    flag[0] = True

            class MyWorkflow(Workflow):
                def __del__(self):
                    flag[1] = True

            with MyWorkflow(DummyLauncher()) as wf:
                u = MyUnit(wf)
                self.assertEqual(len(wf), 3)
                self.assertEqual(u.workflow, wf)

            self.assertEqual(len(wf), 2)
            self.assertEqual(u.workflow, wf)
            self.assertIsInstance(u._workflow_, weakref.ReferenceType)
            del wf
            gc.collect()
            self.assertTrue(flag[1])
            del u
            gc.collect()
            self.assertTrue(flag[0])

        def testDestruction(self):
            flag = [False, False]

            class MyUnit(TrivialUnit):
                def __del__(self):
                    flag[0] = True

            class MyWorkflow(Workflow):
                def __del__(self):
                    flag[1] = True

            wf = MyWorkflow(DummyLauncher())
            u = MyUnit(wf)
            self.assertEqual(len(wf), 3)
            self.assertEqual(u.workflow, wf)
            del u
            del wf
            gc.collect()
            self.assertTrue(flag[0])
            self.assertTrue(flag[1])

    def testPickling(self):
        dl = DummyLauncher()
        wf = Workflow(dl)
        TrivialUnit(wf)
        w2 = pickle.loads(pickle.dumps(wf))
        self.assertEqual(len(w2), len(wf))

    def testRestoredFromSnapshot(self):
        dl = DummyLauncher()
        wf = Workflow(dl)
        self.assertFalse(wf.restored_from_snapshot)
        self.assertFalse(wf.start_point.restored_from_snapshot)
        self.assertIsNone(wf._restored_from_snapshot_)
        wf._restored_from_snapshot_ = True
        self.assertTrue(wf.restored_from_snapshot)
        self.assertTrue(wf.start_point.restored_from_snapshot)
        wf._restored_from_snapshot_ = False
        self.assertFalse(wf.restored_from_snapshot)
        self.assertFalse(wf.start_point.restored_from_snapshot)
        w2 = SnapshotterBase._import_fobj(six.BytesIO(pickle.dumps(wf)))
        self.assertTrue(w2.restored_from_snapshot)
        self.assertTrue(w2.start_point.restored_from_snapshot)
        self.assertTrue(w2._restored_from_snapshot_)
        w2.end_point.link_from(w2.start_point)
        w2.workflow = dl
        w2.initialize()
        self.assertFalse(w2.restored_from_snapshot)
        self.assertFalse(w2.start_point.restored_from_snapshot)
        self.assertIsNone(w2._restored_from_snapshot_)
        w2.link_from(wf)
        wf.end_point.link_from(w2)
        w2.workflow = wf
        self.assertFalse(w2.restored_from_snapshot)
        self.assertFalse(wf.restored_from_snapshot)
        wf._restored_from_snapshot_ = True
        self.assertTrue(w2.restored_from_snapshot)
        self.assertTrue(w2.start_point.restored_from_snapshot)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testItarator']
    unittest.main()
