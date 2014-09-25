"""
Created on Sep 25, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from time import perf_counter


def timeit(function, *args, **kwargs):
    ts = perf_counter()
    res = function(*args, **kwargs)
    return res, perf_counter() - ts
