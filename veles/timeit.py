"""
Created on Sep 25, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from time import clock


def timeit(function, *args, **kwargs):
    ts = clock()
    res = function(*args, **kwargs)
    return res, clock() - ts
