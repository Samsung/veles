"""
Created on Jul 21, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import zope.interface.verify as verify


def _incompat(required, implemented):
    if (len(implemented['required']) > len(required['required']) and
            not required['kwargs']):
        return 'implementation requires too many arguments'
    if ((len(implemented['positional']) < len(required['positional']))
            and not implemented['varargs']):
        return "implementation doesn't allow enough arguments"
    if required['kwargs'] and not implemented['kwargs']:
        return "implementation doesn't support keyword arguments"
    if required['varargs'] and not implemented['varargs']:
        return "implementation doesn't support variable arguments"

verify._incompat = _incompat
