"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 21, 2013

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


import os
import six
import sys
import pdb
import threading
import unittest

from veles.distributable import Pickleable
from veles.pickle2 import setup_pickle_debug, pickle


g_pt = 0


class PickleTest(Pickleable):
    """Pickle test.
    """
    def __init__(self, a="A", b="B", c="C"):
        super(PickleTest, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def init_unpickled(self):
        super(PickleTest, self).init_unpickled()
        global g_pt
        g_pt += 1
        self.c = None


class TestPickle(unittest.TestCase):
    def test_pickle(self):
        # Test for correct behavior of units.Pickleable
        pt = PickleTest(a="AA", c="CC")
        self.assertEqual(g_pt, 1, "Pickle test failed.")
        self.assertEqual("CC", pt.c)
        pt.d = "D"
        pt.h_ = "HH"
        try:
            os.mkdir("cache")
        except OSError:
            pass
        with open("cache/test.pickle", "wb") as fout:
            pickle.dump(pt, fout)
        del pt
        with open("cache/test.pickle", "rb") as fin:
            pt = pickle.load(fin)
        self.assertListEqual([g_pt, pt.d, pt.c, pt.b, pt.a, pt.h_],
                             [2, "D", None, "B", "AA", None],
                             "Pickle test failed.")

    def test_setup_pickle_debug(self):
        stderr = sys.stderr
        sys.stderr = six.StringIO()
        flag = [False]

        def set_trace():
            # nonlocal flag raises SyntaxError on Python 2.7
            flag[0] = True

        pdb.set_trace = set_trace
        dump = pickle.dump
        dumps = pickle.dumps
        load = pickle.load
        loads = pickle.loads

        setup_pickle_debug()
        pickle.dumps(threading.Lock())
        self.assertTrue(flag[0])

        def recover_set_trace():
            print("pdb.set_trace()")

        pdb.set_trace = recover_set_trace
        pickle.dump = dump
        pickle.dumps = dumps
        pickle.load = load
        pickle.loads = loads

        err = sys.stderr.getvalue()
        sys.stderr = stderr
        self.assertTrue(err.find("Pickling failure") >= 0)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test']
    unittest.main()
