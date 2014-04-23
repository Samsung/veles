"""
Created on Apr 23, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import unittest

from veles.mutable import Bool, LinkableAttribute


class A(object):
    pass


class B(object):
    pass


class Test(unittest.TestCase):
    def testBool(self):
        a = Bool()
        self.assertFalse(a)
        b = Bool(True)
        self.assertTrue(b)
        c = a | b
        self.assertTrue(c)
        b << False
        self.assertFalse(a)
        self.assertFalse(b)
        self.assertFalse(c)
        c = a & b
        self.assertFalse(c)
        a << True
        self.assertTrue(a)
        self.assertFalse(b)
        self.assertFalse(c)
        b << True
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertTrue(c)
        c = a ^ b
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertFalse(c)
        a << False
        self.assertFalse(a)
        self.assertTrue(c)
        b << False
        self.assertFalse(a)
        self.assertFalse(b)
        self.assertFalse(c)
        c = ~a
        self.assertFalse(a)
        self.assertTrue(c)
        a << True
        self.assertTrue(a)
        self.assertFalse(c)
        c = a & ~b
        self.assertTrue(c)

    def testLinkableAttribute(self):
        a = A()
        a.number = 77
        b = B()
        LinkableAttribute(b, "number", (a, "number"))
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


if __name__ == "__main__":
    unittest.main()
