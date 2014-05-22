"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import pickle
import unittest

import veles.units as units


g_pt = 0


class PickleTest(units.Pickleable):
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
        del(pt)
        with open("cache/test.pickle", "rb") as fin:
            pt = pickle.load(fin)
        self.assertListEqual([g_pt, pt.d, pt.c, pt.b, pt.a, pt.h_],
                             [2, "D", None, "B", "AA", None],
                             "Pickle test failed.")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test']
    unittest.main()
