"""
Created on Oct 14, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import six

if six.PY3:
    import lzma  # pylint: disable=W0611
else:
    try:
        from backports import lzma
    except ImportError:
        import warnings

        warnings.warn("Failed to import backports.lzma - LZMA/XZ compression "
                      "will be unavailable.\npip install backports.lzma")


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


if not six.PY3:
    import os
    import sys

    DEBUG_BYTECODE_SUFFIXES = ['.pyc']
    OPTIMIZED_BYTECODE_SUFFIXES = ['.pyo']
    _PYCACHE = '__pycache__'

    def cache_from_source(path, debug_override=None):
        """Given the path to a .py file, return the path to its .pyc/.pyo file.

        The .py file does not need to exist; this simply returns the path to
        the .pyc/.pyo file calculated as if the .py file were imported.  The
        extension will be .pyc unless sys.flags.optimize is non-zero, then it
        will be .pyo.

        If debug_override is not None, then it must be a boolean and is used in
        place of sys.flags.optimize.

        If sys.implementation.cache_tag is None then NotImplementedError is
        raised.

        """
        debug = (not sys.flags.optimize if debug_override is None
                 else debug_override)
        if debug:
            suffixes = DEBUG_BYTECODE_SUFFIXES
        else:
            suffixes = OPTIMIZED_BYTECODE_SUFFIXES
        head, tail = os.path.split(path)
        base_filename, sep, _ = tail.partition('.')
        tag = sys.implementation.cache_tag
        if tag is None:
            raise NotImplementedError('sys.implementation.cache_tag is None')
        filename = ''.join([base_filename, sep, tag, suffixes[0]])
        return os.path.join(head, _PYCACHE, filename)
