"""
Created on Apr 15, 2013

Data formats for connectors.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import numpy
import os
import threading
import opencl4py as cl

from veles.config import root
import veles.error as error
import veles.opencl_types as opencl_types
import veles.units as units


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


def assert_addr(a, b):
    """Raises an exception if addresses of the supplied arrays differ.
    """
    if a.__array_interface__["data"][0] != b.__array_interface__["data"][0]:
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


def real_normalize(a):
    """Normalizes real array to [-1, 1] in-place.
    """
    a -= a.min()
    m = a.max()
    if m:
        a /= m
        a *= 2.0
        a -= 1.0


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

    logging.info("%f %f %f %f" % (IMul.min(), IMul.max(),
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


class Vector(units.Pickleable):
    """Container class for numpy array backed by OpenCL buffer.

    Attributes:
        device: OpenCL device.
        v: numpy array.
        v_: OpenCL buffer mapped to v.
        supposed_maxvle: supposed maximum element value.
        map_arr_: pyopencl map object.
        map_flags: flags of the current map.

    Example of how to use:
        1. Construct an object:
            a = formats.Vector()
        2. Connect units be data:
            u2.b = u1.a
        3. Initialize numpy array:
            a.v = numpy.zeros(...)
        4. Initialize an object with device:
            a.initialize(device)
        5. Set OpenCL buffer as kernel parameter:
            krn.set_arg(0, a.v_)

    Example of how to update vector:
        1. Call a.map_write() or a.map_invalidate()
        2. Update a.v
        3. Call a.unmap() before executing OpenCL kernel
    """
    def __init__(self):
        super(Vector, self).__init__()
        self.device = None
        self.v = None
        self.supposed_maxvle = 1.0

    def init_unpickled(self):
        super(Vector, self).init_unpickled()
        self.v_ = None
        self.map_arr_ = None
        self.map_flags = 0
        self.lock_ = threading.Lock()

    def __getstate__(self):
        """Get data from OpenCL device before pickling.
        """
        if (self.device is not None and self.device.pid_ == os.getpid()):
            self.map_read()
        return super(Vector, self).__getstate__()

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
        if self.v is None or self.v_ is not None:
            return
        if device is not None:
            self.device = device
        if self.device is None:
            return
        converted_dtype = self._converted_dtype(self.v.dtype)
        if converted_dtype in (opencl_types.dtypes[root.common.dtype],
                               opencl_types.dtypes[root.common.precision_type]):
            self.v = self.v.astype(converted_dtype)
        self.v = cl.realign_array(self.v, self.device.device_info.memalign,
                                  numpy)
        self.v_ = self.device.queue_.context.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, ravel(self.v))

    def initialize(self, device=None):
        self.lock_.acquire()
        self._initialize(device)
        self.lock_.release()

    def _map(self, flags):
        if self.device is None:
            return
        if self.map_arr_ is not None:
            # already mapped properly, nothing to do
            if (self.map_flags != cl.CL_MAP_READ or flags == cl.CL_MAP_READ):
                return
            self._unmap()
        if (flags == cl.CL_MAP_WRITE_INVALIDATE_REGION and
            self.device.device_info.version < 1.1999):
            # 'cause available only starting with 1.2
            flags = cl.CL_MAP_WRITE
        ev, self.map_arr_ = self.device.queue_.map_buffer(self.v_, flags,
                                                          self.v.nbytes)
        del ev
        self.map_flags = flags

    def _unmap(self):
        if self.map_arr_ is None:
            return
        ev = self.device.queue_.unmap_buffer(self.v_, self.map_arr_)
        ev.wait()
        self.map_arr_ = None
        self.map_flags = 0

    def map_read(self):
        self.lock_.acquire()
        self._map(cl.CL_MAP_READ)
        self.lock_.release()

    def map_write(self):
        self.lock_.acquire()
        self._map(cl.CL_MAP_WRITE)
        self.lock_.release()

    def map_invalidate(self):
        self.lock_.acquire()
        self._map(cl.CL_MAP_WRITE_INVALIDATE_REGION)
        self.lock_.release()

    def unmap(self):
        self.lock_.acquire()
        self._unmap()
        self.lock_.release()

    def __len__(self):
        """To enable [] operator.
        """
        return self.v.size

    def __getitem__(self, key):
        """To enable [] operator.
        """
        return self.v[key]

    def __setitem__(self, key, value):
        """To enable [] operator.
        """
        self.v[key] = value

    def reset(self):
        """Sets buffers to None
        """
        self.lock_.acquire()
        self._unmap()
        self.v = None
        self.v_ = None
        self.map_flags = 0
        self.lock_.release()

    def __del__(self):
        self.reset()
