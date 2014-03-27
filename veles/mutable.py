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

    def __init__(self, value=False):
        self.__expr = [[None]]
        self << value

    @property
    def expr(self):
        return self.__expr

    def __bool__(self):
        initial = self.expr[0][0]
        value = initial if not callable(initial) else initial()
        for method in self.expr[1:]:
            value = method[1](method[0], value)
        return value

    def __lshift__(self, value):
        if len(self.__expr) > 1:
            raise RuntimeError("Derived expressions cannot be assigned to.")
        if isinstance(value, Bool):
            self.__expr = copy(value.__expr)
            return
        if isinstance(value, bool) or callable(value):
            self.__expr[0][0] = value
            return
        raise TypeError("Value must be a boolean value or a function")

    def __derive(name):
        def wrapped(self):
            return getattr(bool(self), name)()
        return wrapped

    __int__ = __derive("__int__")
    __repr__ = __derive("__repr__")
    __str__ = __derive("__str__")

    __derive = staticmethod(__derive)

    def __binary_op(func):
        method = "_Bool" + func.__name__[:-2]

        def wrapped(self, value):
            if isinstance(value, bool):
                res = func(value)
                if res is not None:
                    return res
            res = Bool(self)
            res.expr.append((value, getattr(Bool, method)))
            return res
        return wrapped

    @staticmethod
    def __or(self, value):
        return value or bool(self)

    @__binary_op
    def __or__(self, value):
        return Bool(True) if value else self

    @staticmethod
    def __and(self, value):
        return value and bool(self)

    @__binary_op
    def __and__(self, value):
        return Bool(False) if not value else self

    @staticmethod
    def __xor(self, value):
        return value != bool(self)

    @__binary_op
    def __xor__(self, value):
        return None

    @staticmethod
    def __invert(self, value):
        return not value

    def __invert__(self):
        res = Bool(self)
        res.expr.append((None, Bool.__invert))
        return res

    __binary_op = staticmethod(__binary_op)

    def __getstate__(self):
        state = {"__expr": [self.expr[0]]}
        es = state["__expr"]
        for expr in self.expr[1:]:
            es.append((expr[0], expr[1].__name__,
                       marshal.dumps(expr[1].__code__)))
        return state

    def __setstate__(self, state):
        es = self.__expr = [state["__expr"][0]]
        for expr in state["__expr"][1:]:
            func_code = marshal.loads(expr[2])
            es.append((expr[0],
                       types.FunctionType(func_code, globals(), expr[1])))
