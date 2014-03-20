"""
Created on Sep 10, 2013

Image Scale.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import os

from veles.config import root
import veles.error as error
from ctypes import POINTER, c_byte, c_int, cdll


handle = cdll.LoadLibrary(os.path.join(root.common.veles_dir,
                                       "veles/libyuv.so"))
handle.ScalePlane.argtypes = [POINTER(c_byte), c_int, c_int, c_int,
                              POINTER(c_byte), c_int, c_int, c_int, c_int]


BILINEAR = 1
BOX = 2


resize_count = 0
asitis_count = 0


def resize(a, width, height, interpolation=BILINEAR, require_copy=False):
    if a.dtype != numpy.uint8:
        raise error.ErrBadFormat("a.dtype should be numpy.uint8")
    if a.shape[1] == width and a.shape[0] == height:
        global asitis_count
        asitis_count += 1
        return a.copy() if require_copy else a
    b = numpy.zeros([height, width], dtype=a.dtype)
    handle.ScalePlane(a.ctypes.data_as(POINTER(c_byte)),
                      a.shape[1], a.shape[1], a.shape[0],
                      b.ctypes.data_as(POINTER(c_byte)),
                      b.shape[1], b.shape[1], b.shape[0], interpolation)
    global resize_count
    resize_count += 1
    return b
