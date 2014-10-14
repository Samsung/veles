"""
Created on Oct 14, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


def from_none(exc):
    """Emulates raise ... from None (PEP 409) on older Python-s
    """
    try:
        exc.__cause__ = None
    except AttributeError:
        exc.__context__ = None
    return exc
