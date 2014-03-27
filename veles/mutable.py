"""
Created on Mar 26, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from copy import copy
import marshal
import types


class Bool(object):
    """
    A mutable boolean value.

    Supported operations:
        a | b       <=>  a or b
        a & b       <=>  a and b
        a ^ b       <=>  a != b
        ~a          <=>  not a
        a << True   <=>  a = True
        a << False  <=>  a = False
        a << b      <=>  a = copy(b)
        a = b       <=>  a is b
        a << function() -> False|True
        bool(a), int(a)
    """

    def __init__(self, value):
        self.__expr = [[None]]
        self.__lshift__(value)

    def __bool__(self):
        value = None
        for method in self.__expr:
            value = method[0](value)
        return value

    @property
    def expr(self):
        return self.__expr

    def __derive(name):
        def wrapped(self):
            return getattr(bool(self), name)()
        return wrapped

    __int__ = __derive("__int__")
    __repr__ = __derive("__repr__")
    __str__ = __derive("__str__")

    __derive = staticmethod(__derive)

    def __lshift__(self, value):
        if len(self.__expr) > 1:
            raise RuntimeError("Derived expressions cannot be assigned to.")
        if isinstance(value, Bool):
            self.__expr = copy(value.__expr)
            return
        if isinstance(value, bool):
            self.__expr[0][0] = Bool.__true if value else Bool.__false
            return
        if callable(value):
            self.__expr[0][0] = value
        raise TypeError("Value must be a boolean value or a function")

    @staticmethod
    def __true(value=None):
        return True

    @staticmethod
    def __false(value=None):
        return False

    def __binary_op(func):
        method = "_Bool" + func.__name__[:-2]

        def wrapped(self, value=None):
            if isinstance(value, bool):
                res = func(value)
                if res is not None:
                    return res
                else:
                    res.expr.append([getattr(Bool(value), method)])
            res = Bool(self)
            res.expr.append([getattr(value or self, method)])
            return res
        return wrapped

    def __or(self, value):
        return value or self.__bool__()

    @__binary_op
    def __or__(self, value):
        return True if value else self

    def __and(self, value):
        return value and self.__bool__()

    @__binary_op
    def __and__(self, value):
        return False if not value else self

    def __xor(self, value):
        return value != self.__bool__()

    @__binary_op
    def __xor__(self, value):
        return None

    def __invert(self, value):
        return not value

    @__binary_op
    def __invert__(self):
        pass

    __binary_op = staticmethod(__binary_op)

    def __getstate__(self):
        state = {"__expr": []}
        es = state["__expr"]
        for expr in self.expr:
            es.append((expr[0].__name__, marshal.dumps(expr[0].__code__)))
        return state

    def __setstate__(self, state):
        es = self.__expr = []
        for expr in state["__expr"]:
            func_code = marshal.loads(expr[1])
            es.append([types.FunctionType(func_code, globals(), expr[0])])
