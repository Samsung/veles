"""
Created on May 28, 2013

Global configuration variables.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy


#: Supported float types as OpenCL => numpy dictionary.
dtypes = {"float": numpy.float32, "double": numpy.float64}

#: Current number type
#dtype = "float"
dtype = "double"

#: CL defines
cl_defines = {"float": "#define dtype float",
              "double": "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
                        "#define dtype double"}

#: optional/inline.py argument types
inline_types = {"float": "f", "double": "d"}

#: Supported int types as OpenCL => numpy dictionary.
itypes = {"char": numpy.int8, "short": numpy.int16, "int": numpy.int32,
          "long": numpy.int64,
          "uchar": numpy.uint8, "ushort": numpy.uint16, "uint": numpy.uint32,
          "ulong": numpy.uint64}


# Directories
import os

#: Directory with config.py itself
this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."

#: Directory for cache
cache_dir = ("%s/../cache" % (this_dir))
try:
    os.mkdir(cache_dir)
except OSError:
    pass

#: Directory for OpenCL source files
cl_dir = ("%s/../Znicz/cl" % (this_dir))

#: Directory where to save snapshots
snapshot_dir = ("%s/../snapshots" % (this_dir))
try:
    os.mkdir(snapshot_dir)
except OSError:
    pass

#: Directory where functional tests large datasets reside.
test_dataset_root = "/data/veles"


def get_itype_from_size(size, signed=True):
    """Get integer type from size.
    """
    if signed:
        if size < 128:
            return "char"
        if size < 32768:
            return "short"
        if size < 2147483648:
            return "int"
        return "long"
    if size < 256:
        return "uchar"
    if size < 65536:
        return "ushort"
    if size < 4294967296:
        return "uint"
    return "ulong"
