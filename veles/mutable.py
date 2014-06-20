"""
Created on Mar 26, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from copy import copy
import inspect
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
        a <<= True  <=>  a = True
        a <<= False <=>  a = False
        a <<= b     <=>  a = bool(b)
        a = b       <=>  a is b
        a <<= function() -> False|True
        a(b)        <=>  a = copy(b)
        bool(a), int(a)
    """

    def __init__(self, value=False):
        self.__expr = [[None]]
        self.__influences = {self}
        self.on_true = None
        self.on_false = None
        if not isinstance(value, Bool):
            self <<= value
        else:
            self.__expr = copy(value.__expr)
            value.__influences.add(self)

    @property
    def expr(self):
        return self.__expr

    def __bool__(self):
        initial = self.expr[0][0]
        if not callable(initial):
            value = initial
        else:
            value = initial()  # pylint: disable=E1102
        for method in self.expr[1:]:
            value = method[1](method[0], value)
        return value

    def __nonzero__(self):
        return self.__bool__()

    def __ilshift__(self, value):
        if len(self.__expr) > 1:
            raise RuntimeError("Derived expressions cannot be assigned to.")
        if isinstance(value, Bool):
            self.__expr[0][0] = bool(value)
            self.touch()
            return self
        if isinstance(value, bool) or callable(value):
            self.__expr[0][0] = value
            self.touch()
            return self
        raise TypeError("Value must be a boolean value or a function")

    def __derive(name):
        def wrapped(self):
            return getattr(bool(self), name)()
        wrapped.__name__ = name + '_derived'
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
            value.__influences.add(res)
            res.expr.append((value, getattr(Bool, method)))
            return res
        wrapped.__name__ = method + '_binary_op'
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
        self.on_true = None
        self.on_false = None
        self.__influences = {self}

    def touch(self):
        """
        Raises events on influenced Bool-s.
        Crashes badly on Python 3.4.
        """
        influenced = copy(self.__influences)
        pending = copy(influenced)
        while len(pending):
            item = pending.pop()
            new = item.__influences - influenced
            influenced.update(new)
            pending.update(new)

        for b in influenced:
            if b.on_true is None and b.on_false is None:
                continue
            if b:
                if b.on_true is not None:
                    b.on_true(b)
            else:
                if b.on_false is not None:
                    b.on_false(b)

    def unref(self, item):
        self.__influences.remove(item)


class LinkableAttribute(object):
    """Data descriptor which allows referencing an attribute belonging to
    an other class instance. Internally, it operates with "pointers" modeled
    with tuples (obj, attr_name).

    The usage is the following: at any time in the program you can call
    LinkableAttribute(class_instance, string_with_attr_name, value).
    It will create a static class attribute 'string_with_attr_name' and
    set LinkableAttribute() to it. That instance of LinkableAttribute saves the
    name of the attribute with prepended double underscore "__" as well as the
    unmodified name.
    The "pointer" can be replaced with anything else, resulting in passing
    the attribute value "by value", as usual in Python.

    Applying LinkableAttribute() on the same attribute is safe.
    """

    def __new__(cls, *args, **kwargs):
        # check if we already have an instance of LinkableAttribute assigned to
        # the attribute args[1] of the type type(args[0])

        # if we unpickle this class unpickle method will try
        # to create instance with args = ()
        if len(args) < 2:
            return super(LinkableAttribute, cls).__new__(cls)
        instance = getattr(type(args[0]), args[1], None)
        if not isinstance(instance, LinkableAttribute):
            return super(LinkableAttribute, cls).__new__(cls)
        # updating the attribute value since the object already exists and thus
        # we are ignoring __init__()
        setattr(*args[:3])
        LinkableAttribute._set_option(instance, 3, "two_way", *args, **kwargs)
        LinkableAttribute._set_option(instance, 4, "assignment_guard",
                                      *args, **kwargs)

    @staticmethod
    def _set_option(instance, index, name, *args, **kwargs):
        """Called from __new__, this method is a convenience ctor option setter
        """
        if len(args) > index:
            setattr(instance, name, args[index])
        else:
            value = kwargs.get(name)
            if value is not None:
                setattr(instance, name, value)

    def __init__(self, obj, name, value, two_way=False, assignment_guard=True):
        if obj is None:
            raise UnboundLocalError(
                self.__class__,
                "can not be created without an instance to bind: instance=",
                obj, "name=", name, "value=", value)
        self.two_way = two_way
        self.assignment_guard = assignment_guard
        # getting here means that passed the instance check in  __new__
        # real name of the attribute
        self.real_attribute_name = '__' + name
        # original name without underscores is used in __get__ to find
        # the class attribute faster
        self.exposed_attribute_name = name

        # assign the attribute of the hosting class
        setattr(type(obj), name, self)

        # assign the attribute value to "obj"
        setattr(obj, self.real_attribute_name, value)

    def __get__(self, obj, objtype):
        # since this method can be applied to get the attribute of the Class
        # (not just it's instance) we check whether obj is None; see __new__
        if obj is None:
            return objtype.__getattribute__(objtype,
                                            self.exposed_attribute_name)
        # get the reference to the attribute value
        pointer = getattr(obj, self.real_attribute_name)
        # dereference it
        return getattr(*pointer)

    def __set__(self, obj, value):
        if not LinkableAttribute.__is_reference__(value):
            if self.two_way:
                # update the referenced attribute value and return
                pointer = getattr(obj, self.real_attribute_name)
                setattr(pointer[0], pointer[1], value)
                return
            elif self.assignment_guard:
                prev_value = getattr(obj, self.real_attribute_name, None)
                if inspect.isclass(obj) or \
                   LinkableAttribute.__is_reference__(prev_value):
                    raise RuntimeError("Attempted to set the value of linked "
                                       "property '%s' in object %s and "
                                       "two_way is switched off." %
                                       (self.exposed_attribute_name, str(obj)))
            else:
                # play the trick with getattr(*pointer) in __get__
                value = (None, '', value)
        else:
            if value[0] == obj and value[1] == self.real_attribute_name:
                raise ValueError("Attempted to set the attribute reference to "
                                 "itself")
        setattr(obj, self.real_attribute_name, value)

    def __delete__(self, obj):
        obj.__delattr__(self.real_attribute_name)

    @staticmethod
    def __is_reference__(value):
        return isinstance(value, tuple) and len(value) == 2 and \
            isinstance(value[0], object) and isinstance(value[1], str)


def link(obj_dst, name_dst, obj_src, name_src, two_way=False):
    """Establishes a link from obj_src's "name_src" attribute to obj_dst's
    "name_dst" one using LinkableAttribute.
    """
    LinkableAttribute(obj_dst, name_dst, (obj_src, name_src), two_way)
