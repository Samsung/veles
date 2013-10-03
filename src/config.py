"""
Created on May 28, 2013

Global configuration variables.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy


#: Supported float types as OpenCL => numpy dictionary.
dtypes = {"float": numpy.float32, "double": numpy.float64,
          "float2": numpy.complex64, "double2": numpy.complex128}

#: Complex type
#c_dtype = "float"  # for real single precision numbers
c_dtype = "double"  # for real numbers
#c_dtype = "double2"  # for complex numbers
#c_dtype = "double4"  # for quad numbers (not implemented)

#: Complex type to real type mapping
dtype_map = {"float": "float", "double": "double",
             "float2": "float", "double2": "double"}

#: Real type
dtype = dtype_map[c_dtype]

#: CL defines
cl_defines = {"float": "#define dtype float\n"
                       "#define c_dtype float\n"
                       "#define sizeof_c_dtype 4",
              "double": "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
                        "#define dtype double\n"
                        "#define c_dtype double\n"
                        "#define sizeof_c_dtype 8",
              "float2": "#define COMPLEX\n"
                        "#define dtype float\n"
                        "#define c_dtype float2\n"
                        "#define sizeof_c_dtype 8",
              "double2": "#define COMPLEX\n"
                         "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
                         "#define dtype double\n"
                         "#define c_dtype double2\n"
                         "#define sizeof_c_dtype 16"}

#: Supported int types as OpenCL => numpy dictionary.
itypes = {"char": numpy.int8, "short": numpy.int16, "int": numpy.int32,
          "long": numpy.int64,
          "uchar": numpy.uint8, "ushort": numpy.uint16, "uint": numpy.uint32,
          "ulong": numpy.uint64}


# Allowable types for automatic conversion for use in OpenCL
# (automatic conversion implemeted in formats.py).
convert_map = {numpy.float32: numpy.float64, numpy.float64: numpy.float32,
               numpy.complex64: numpy.complex128,
               numpy.complex128: numpy.complex64}


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
