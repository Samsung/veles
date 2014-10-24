"""
Created on Oct 14, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import six


def from_none(exc):
    """Emulates raise ... from None (PEP 409) on older Python-s
    """
    try:
        exc.__cause__ = None
    except AttributeError:
        exc.__context__ = None
    return exc

if six.PY3:
    from enum import IntEnum  # pylint: disable=W0611
else:
    class EnumMeta(type):
        def __init__(cls, name, bases, dict):
            super(EnumMeta, cls).__init__(name, bases, dict)
            cls.__slots__ = [k for k in dict.keys() if k != "__metaclass__"]

        def __getitem__(cls, value):
            return cls.__dict__[value]

        def __setattr__(cls, attr, value):
            if attr != "__slots__":
                raise AttributeError("You may not alter enum values")

    class IntEnum(object):
        __metaclass__ = EnumMeta
