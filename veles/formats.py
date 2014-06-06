"""
Created on Apr 15, 2013

Data formats for connectors.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import logging
import numpy
import os
import threading
import opencl4py as cl

import veles.error as error
from veles.distributable import Pickleable


def roundup(num, align):
    d = num % align
    if d == 0:
        return num
    return num + (align - d)


def max_type(num):
    """Returns array converted to supported type with maximum precision.
    """
    if num.dtype in [numpy.complex64, numpy.complex128]:
        return num.astype(numpy.complex128)
    return num.astype(numpy.float64)


def eq_addr(a, b):
    return a.__array_interface__["data"][0] == b.__array_interface__["data"][0]


def assert_addr(a, b):
    """Raises an exception if addresses of the supplied arrays differ.
    """
    if not eq_addr(a, b):
        raise error.ErrBadFormat("Addresses of the arrays are not equal.")


def ravel(a):
    """numpy.ravel() with address check.
    """
    b = a.ravel()
    assert_addr(a, b)
    return b


def reshape(a, shape):
    """numpy.reshape() with address check.
    """
    b = a.reshape(shape)
    assert_addr(a, b)
    return b


def interleave(a):
    """Returns interleaved array.

    Example:
        [10000, 3, 32, 32] => [10000, 32, 32, 3].
    """
    sh = list(a.shape)
    sh.append(sh.pop(-3))
    b = numpy.empty(sh, dtype=a.dtype)
    if len(b.shape) == 4:
        for i in range(sh[-1]):
            b[:, :, :, i] = a[:, i, :, :]
    elif len(b.shape) == 3:
        for i in range(sh[-1]):
            b[:, :, i] = a[i, :, :]
    else:
        raise error.ErrBadFormat("a should be of shape 4 or 3.")
    return b


def real_normalize(a):
    """Normalizes real array to [-1, 1] in-place.
    """
    x_min, x_max = a.min(), a.max()
    a -= x_min
    if x_max > x_min:
        a *= 2. / (x_max - x_min)
        a -= 1.


def complex_normalize(a):
    """Normalizes complex array to unit-circle in-place.
    """
    a -= numpy.average(a)
    m = numpy.sqrt((a.real * a.real + a.imag * a.imag).max())
    if m:
        a /= m


def normalize(a):
    if a.dtype in (numpy.complex64, numpy.complex128):
        return complex_normalize(a)
    return real_normalize(a)


def normalize_mean_disp(a):
    if a.dtype in (numpy.complex64, numpy.complex128):
        return complex_normalize(a)
    mean = numpy.mean(a)
    mi = numpy.min(a)
    mx = numpy.max(a)
    ds = max(mean - mi, mx - mean)
    a -= mean
    if ds:
        a /= ds


def normalize_exp(a):
    if a.dtype in (numpy.complex64, numpy.complex128):
        raise error.ErrNotImplemented()
    a -= a.max()
    numpy.exp(a, a)
    smm = a.sum()
    a /= smm


def normalize_pointwise(a):
    """Normalizes dataset pointwise.
    """
    IMul = numpy.zeros_like(a[0])
    IAdd = numpy.zeros_like(a[0])

    mins = numpy.min(a, 0)
    maxs = numpy.max(a, 0)
    ds = maxs - mins
    zs = numpy.nonzero(ds)

    IMul[zs] = 2.0
    IMul[zs] /= ds[zs]

    mins *= IMul
    IAdd[zs] = -1.0
    IAdd[zs] -= mins[zs]

    logging.getLogger("Loader").debug("%f %f %f %f" % (IMul.min(), IMul.max(),
                                                       IAdd.min(), IAdd.max()))

    return (IMul, IAdd)


def norm_image(a, yuv=False):
    """Normalizes numpy array to interval [0, 255].
    """
    aa = a.astype(numpy.float32)
    if aa.__array_interface__["data"][0] == a.__array_interface__["data"][0]:
        aa = aa.copy()
    aa -= aa.min()
    m = aa.max()
    if m:
        m /= 255.0
        aa /= m
    else:
        aa[:] = 127.5
    if yuv and len(aa.shape) == 3 and aa.shape[2] == 3:
        aaa = numpy.empty_like(aa)
        aaa[:, :, 0:1] = aa[:, :, 0:1] + (aa[:, :, 2:3] - 128) * 1.402
        aaa[:, :, 1:2] = (aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * (-0.34414) +
                          (aa[:, :, 2:3] - 128) * (-0.71414))
        aaa[:, :, 2:3] = aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * 1.772
        numpy.clip(aaa, 0.0, 255.0, aa)
    return aa.astype(numpy.uint8)


class NumDiff(object):
    """Numeric differentiation helper.

    WARNING: it is invalid for single precision float data type.
    """
    def __init__(self):
        self.h = 1.0e-8
        self.points = (2.0 * self.h, self.h, -self.h, -2.0 * self.h)
        self.coeffs = numpy.array([-1.0, 8.0, -8.0, 1.0],
                                  dtype=numpy.float64)
        self.divizor = 12.0 * self.h
        self.errs = numpy.zeros_like(self.points)

    @property
    def derivative(self):
        return (self.errs * self.coeffs).sum() / self.divizor


class Vector(Pickleable):
    """Container class for numpy array backed by OpenCL buffer.

    Arguments:
        data(:class:`numpy.ndarray`): `mem` attribute will be assigned to this

    Attributes:
        device: OpenCL device.
        mem: property for numpy array.
        devmem: OpenCL buffer mapped to mem.
        _mem: numpy array.
        supposed_maxvle: supposed maximum element value.
        map_arr_: pyopencl map object.
        map_flags: flags of the current map.

    Example of how to use:
        1. Construct an object:
            a = formats.Vector()
        2. Connect units be data:
            u2.b = u1.a
        3. Initialize numpy array:
            a.mem = numpy.zeros(...)
        4. Initialize an object with device:
            a.initialize(device)
        5. Set OpenCL buffer as kernel parameter:
            krn.set_arg(0, a.devmem)

    Example of how to update vector:
        1. Call a.map_write() or a.map_invalidate()
        2. Update a.mem
        3. Call a.unmap() before executing OpenCL kernel
    """
    def __init__(self, data=None):
        super(Vector, self).__init__()
        self._mem = data
        self.device = None
        self.supposed_maxvle = 1.0

    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, value):
        if self.devmem is not None and not eq_addr(self._mem, value):
            raise error.ErrExists("OpenCL buffer already assigned, "
                                  "call reset() beforehand.")
        self._mem = value

    def init_unpickled(self):
        super(Vector, self).init_unpickled()
        self.devmem = None
        self.map_arr_ = None
        self.map_flags = 0
        self.lock_ = threading.Lock()

    def threadsafe(fn):
        def wrapped(self, *args, **kwargs):
            with self.lock_:
                res = fn(self, *args, **kwargs)
            return res
        return wrapped

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if self.device is not None and self.device.pid_ == os.getpid():
            self.map_read()
        state = super(Vector, self).__getstate__()
        state['devmem'] = None
        return state

    def __bool__(self):
        return self._mem is not None and len(self._mem) > 0

    def __nonzero__(self):
        return self.__bool__()

    def __lshift__(self, value):
        self._mem = value

    def __rlshift__(self, other):
        other.extend(self._mem)

    def __len__(self):
        """To enable [] operator.
        """
        return self._mem.size

    def __del__(self):
        self.reset()

    def __getitem__(self, key):
        """To enable [] operator.
        """
        return self._mem[key]

    def __setitem__(self, key, value):
        """To enable [] operator.
        """
        self._mem[key] = value

    def _converted_dtype(self, dtype):
        if dtype == numpy.float32:
            return numpy.float64
        if dtype == numpy.float64:
            return numpy.float32
        if dtype == numpy.complex64:
            return numpy.complex128
        if dtype == numpy.complex128:
            return numpy.complex64
        return None

    def _initialize(self, device):
        if self._mem is None or self.devmem is not None:
            return
        if device is not None:
            self.device = device
        if self.device is None:
            return
        self._mem = cl.realign_array(self._mem,
                                     self.device.device_info.memalign,
                                     numpy)
        self.devmem = self.device.queue_.context.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, ravel(self._mem))

    @threadsafe
    def initialize(self, device=None):
        self._initialize(device)

    def _map(self, flags):
        if self.device is None:
            return
        if self.map_arr_ is not None:
            # already mapped properly, nothing to do
            if self.map_flags != cl.CL_MAP_READ or flags == cl.CL_MAP_READ:
                return
            self._unmap()
        if (flags == cl.CL_MAP_WRITE_INVALIDATE_REGION and
                self.device.device_info.version < 1.1999):
            # 'cause available only starting with 1.2
            flags = cl.CL_MAP_WRITE
        ev, self.map_arr_ = self.device.queue_.map_buffer(self.devmem, flags,
                                                          self._mem.nbytes)
        if (int(cl.ffi.cast("size_t", self.map_arr_)) !=
                self._mem.__array_interface__["data"][0]):
            raise error.ErrOpenCL("map_buffer returned different pointer")
        del ev
        self.map_flags = flags

    def _unmap(self):
        if self.map_arr_ is None:
            return
        # Workaround Python 3.4.0 incorrect destructor order call bug
        if self.device.queue_.handle is None:
            logging.getLogger("Vector").warning(
                "OpenCL device queue is None but Vector devmem was not "
                "explicitly unmapped.")
        else:
            ev = self.device.queue_.unmap_buffer(self.devmem, self.map_arr_)
            ev.wait()
        self.map_arr_ = None
        self.map_flags = 0

    @threadsafe
    def map_read(self):
        self._map(cl.CL_MAP_READ)

    @threadsafe
    def map_write(self):
        self._map(cl.CL_MAP_WRITE)

    @threadsafe
    def map_invalidate(self):
        self._map(cl.CL_MAP_WRITE_INVALIDATE_REGION)

    @threadsafe
    def unmap(self):
        self._unmap()

    @threadsafe
    def reset(self):
        """Sets buffers to None
        """
        self._unmap()
        self.devmem = None
        self._mem = None
        self.map_flags = 0

    threadsafe = staticmethod(threadsafe)
