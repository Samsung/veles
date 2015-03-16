"""
Created on Apr 23, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import unittest

from veles.mutable import Bool, LinkableAttribute
from veles.pickle2 import pickle


class A(object):
    pass


class B(object):
    pass


class C(object):
    def __init__(self):
        self.number = 255


class Test(unittest.TestCase):
    def set_flag(self, b):
        b.flag = True

    def testBool(self):
        a = Bool()
        self.assertFalse(a)
        b = Bool(True)
        self.assertTrue(b)
        c = a | b
        self.assertTrue(c)
        b <<= False
        self.assertFalse(a)
        self.assertFalse(b)
        self.assertFalse(c)
        c = a & b
        self.assertFalse(c)
        c.flag = False

        c.on_false = self.set_flag
        a <<= True
        self.assertTrue(a)
        self.assertFalse(b)
        self.assertFalse(c)
        self.assertTrue(c.flag)
        c.on_false = None
        c.on_true = self.set_flag
        b <<= True
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertTrue(c)
        self.assertTrue(c.flag)
        a.unref(c)
        b.unref(c)
        c.flag = False
        c.on_false = self.set_flag
        c.on_true = None
        b <<= False
        self.assertFalse(c.flag)
        b <<= True
        c = a ^ b
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertFalse(c)
        a <<= False
        self.assertFalse(a)
        self.assertTrue(c)
        b <<= False
        self.assertFalse(a)
        self.assertFalse(b)
        self.assertFalse(c)
        c = ~a
        self.assertFalse(a)
        self.assertTrue(c)
        a <<= True
        self.assertTrue(a)
        self.assertFalse(c)
        c = a & ~b
        self.assertTrue(c)

    def testBoolPickling(self):
        a = Bool()
        self.assertFalse(a)
        b = Bool(True)
        self.assertTrue(b)
        c = a | b
        c.on_true = self.set_flag
        c2 = pickle.loads(pickle.dumps(c))
        self.assertEqual(None, c2.on_true)

    def testLinkableAttribute(self):
        a = A()
        a.number = 77
        b = B()
        LinkableAttribute(b, "number", (a, "number"), assignment_guard=False)
        # link(b, "number", a, "number")
        self.assertEqual(77, a.number)
        self.assertEqual(77, b.number)
        a.number = 100
        self.assertEqual(100, a.number)
        self.assertEqual(100, b.number)
        b.number = 40
        self.assertEqual(100, a.number)
        self.assertEqual(40, b.number)
        LinkableAttribute(b, "number", (a, "number"), True)
        b.number = 77
        self.assertEqual(77, a.number)
        self.assertEqual(77, b.number)
        self.assertRaises(ValueError,
                          LinkableAttribute(b, "number", (b, "number")))

    def testLinkableAttribute100(self):
        a = A()
        a.number = 77
        b = []
        bb = []
        bbb = []
        for i in range(100):
            b.append(B())
            LinkableAttribute(b[i], "number", (a, "number"))
            bb.append(B())
            LinkableAttribute(bb[i], "number", (b[i], "number"))
            bbb.append(B())
            LinkableAttribute(bbb[i], "number",
                              (bbb[i - 1] if i else bb[i], "number"))
        a.number = 123
        for i in range(100):
            self.assertEqual(b[i].number, 123)
            self.assertEqual(bb[i].number, 123)
            self.assertEqual(bbb[i].number, 123)

    def testLinkableAttributeConstructorAssignment(self):
        a = A()
        a.number = 77
        c = C()
        LinkableAttribute(c, "number", (a, "number"))
        c2 = C()  # exception should not be here
        LinkableAttribute(c2, "number", (a, "number"))

    def testLinkableAttributeConstructorAssignment100(self):
        a = A()
        a.number = 77
        c = []
        cc = []
        ccc = []
        for i in range(100):
            c.append(C())
            LinkableAttribute(c[i], "number", (a, "number"))
            cc.append(C())
            LinkableAttribute(cc[i], "number", (c[i], "number"))
            ccc.append(B())
            LinkableAttribute(ccc[i], "number",
                              (ccc[i - 1] if i else cc[i], "number"))
        a.number = 123
        for i in range(100):
            self.assertEqual(c[i].number, 123)
            self.assertEqual(cc[i].number, 123)
            self.assertEqual(ccc[i].number, 123)

    def testLinkableAttributeOneWay(self):
        class AC(object):
            pass

        class BC(object):
            pass

        a = AC()
        a.number = 77
        b = BC()
        b.number = 88
        LinkableAttribute(b, "number", (a, "number"))
        self.assertEqual(b.number, 77)
        try:
            b.number = 99
            err = True
        except RuntimeError:
            err = False
        self.assertFalse(err, "Two way assignment disabled")

    def testLinkablePickling(self):
        a = A()
        a.number = 77
        b = B()
        LinkableAttribute(b, "number", (a, "number"))
        new_a, new_b = pickle.loads(pickle.dumps((a, b)))
        self.assertEqual(new_b.number, 77)
        self.assertEqual(new_a.number, 77)
        new_a.number = 100
        self.assertEqual(new_b.number, 100)

if __name__ == "__main__":
    unittest.main()
