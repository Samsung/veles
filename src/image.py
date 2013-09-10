"""
Created on Sep 10, 2013

Image Scale.

@author: ajk
"""
import numpy
import config
import error
from ctypes import POINTER, c_byte, c_int, cdll


handle = cdll.LoadLibrary("%s/libyuv.so" % (config.this_dir))
handle.ScalePlane.argtypes = [POINTER(c_byte), c_int, c_int, c_int,
                              POINTER(c_byte), c_int, c_int, c_int, c_int]


BILINEAR = 1
BOX = 2


def resize(a, width, height, interpolation=BILINEAR):
    if a.dtype != numpy.uint8:
        raise error.ErrBadFormat("a.dtype should be numpy.uint8")
    b = numpy.zeros([height, width], dtype=a.dtype)
    handle.ScalePlane(a.ctypes.data_as(POINTER(c_byte)),
                      a.shape[1], a.shape[1], a.shape[0],
                      b.ctypes.data_as(POINTER(c_byte)),
                      b.shape[1], b.shape[1], b.shape[0], interpolation)
    return b
