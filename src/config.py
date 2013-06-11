"""
Created on May 28, 2013

Global configuration variables.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy


# Supported number types as OpenCL => numpy dictionary.
dtypes = {"float": numpy.float32, "double": numpy.float64}

# Current number type
dtype = "float"
#dtype = "double"

# CL pragmas
pragmas = {"float": "",
           "double": "#pragma OPENCL EXTENSION cl_khr_fp64: enable"}
