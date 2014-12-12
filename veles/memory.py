"""
Created on Apr 15, 2013

Data formats for connectors.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
import cuda4py as cu
import logging
import numpy
import six
import os
import threading
import opencl4py as cl

from veles.compat import from_none
import veles.error as error
from veles.distributable import Pickleable
from veles.backends import Device


class WatcherMeta(type):
    def __init__(cls, *args, **kwargs):
        super(WatcherMeta, cls).__init__(*args, **kwargs)
        cls._mutex = threading.Lock()
        cls._mem_in_use = 0
        cls._max_mem_in_use = 0

    def __enter__(cls):
        cls._mutex.acquire()

    def __exit__(cls, *args, **kwargs):
        cls._mutex.release()

    def threadsafe(method):
        def wrapped(cls, *args, **kwargs):
            with cls:
                return method(cls, *args, **kwargs)

        return wrapped

    @property
    def mem_in_use(cls):
        return cls._mem_in_use

    @property
    def max_mem_in_use(cls):
        return cls._max_mem_in_use

    @threadsafe
    def reset_counter(cls):
        cls._max_mem_in_use = cls._mem_in_use

    @threadsafe
    def __iadd__(cls, other):
        cls._mem_in_use += other
        cls._max_mem_in_use = max(cls._mem_in_use, cls._max_mem_in_use)
        return cls

    @threadsafe
    def __isub__(cls, other):
        cls._mem_in_use -= other
        return cls

    threadsafe = staticmethod(threadsafe)


@six.add_metaclass(WatcherMeta)
class Watcher(object):
    pass


def roundup(num, align):
    d = num % align
    if d == 0:
        return num
    return num + (align - d)


def max_type(num):
    """Returns array converted to supported type with maximum precision.
    """
    return num.astype(numpy.float64)


def eq_addr(a, b):
    return a.__array_interface__["data"][0] == b.__array_interface__["data"][0]


def assert_addr(a, b):
    """Raises an exception if addresses of the supplied arrays differ.
    """
    if not eq_addr(a, b):
        raise error.BadFormatError("Addresses of the arrays are not equal.")


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
        raise error.BadFormatError("a should be of shape 4 or 3.")
    return b


def normalize(a):
    """Normalizes array to [-1, 1] in-place.
    """
    a -= a.min()
    mx = a.max()
    if mx:
        a /= mx * 0.5
        a -= 1.0


def normalize_mean_disp(a):
    mean = numpy.mean(a)
    mi = numpy.min(a)
    mx = numpy.max(a)
    ds = max(mean - mi, mx - mean)
    a -= mean
    if ds:
        a /= ds


def normalize_exp(a):
    a -= a.max()
    numpy.exp(a, a)
    smm = a.sum()
    a /= smm


def normalize_pointwise(a):
    """Normalizes dataset pointwise to [-1, 1].
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
    """Container class for numpy array backed by GPU buffer.

    Arguments:
        data(:class:`numpy.ndarray`): `mem` attribute will be assigned to this

    Attributes:
        device: Device object.
        mem: associated numpy array.
        devmem: GPU buffer mapped to mem.
        max_supposed: supposed maximum element value.
        map_flags: flags of the current map.
        map_arr_: address of the mapping if any exists.

    Example of how to use:
        1. Construct an object:
            a = Vector()
        2. Connect units be data:
            u2.b = u1.a
        3. Initialize numpy array:
            a.mem = numpy.zeros(...)
        4. Initialize an object with unit:
            a.initialize(unit)
        5. Set OpenCL buffer as kernel parameter:
            krn.set_arg(0, a.devmem)

    Example of how to update vector:
        1. Call a.map_write() or a.map_invalidate()
        2. Update a.mem
        3. Call a.unmap() before executing OpenCL kernel
    """
    backend_methods = ("map_read", "map_write", "map_invalidate", "unmap",
                       "realign_mem", "create_devmem")

    def __init__(self, data=None):
        super(Vector, self).__init__()
        self._device = None
        self._mem = data
        self._max_value = 1.0

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._reset(False)
        if device is None:
            self._unset_device()
            return
        if not isinstance(device, Device):
            raise TypeError(
                "device must be an instance of veles.opencl.Device, got %s" %
                device)
        self._device = device
        for suffix in Vector.backend_methods:
            setattr(self, "_backend_" + suffix + "_",
                    getattr(self, device.backend_name + "_" + suffix))

    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, value):
        if self.devmem is not None and not eq_addr(self._mem, value):
            raise error.AlreadyExistsError("OpenCL buffer already assigned, "
                                           "call reset() beforehand.")
        self._mem = value

    @property
    def max_supposed(self):
        """
        :return: The supposed maximal value in the contained array. It is NOT
        updated automatically and default to 1. To get the actual maximal
        value, use max().
        """
        return self._max_value

    @max_supposed.setter
    def max_supposed(self, value):
        try:
            1.0 + value
        except TypeError:
            raise from_none(TypeError(
                "max_value must be set to floating point number"))
        self._max_value = value

    @property
    def size(self):
        return self.mem.size

    @property
    def nbytes(self):
        return self.mem.nbytes

    @property
    def itemsize(self):
        return self.mem.itemsize

    @property
    def shape(self):
        return self.mem.shape

    @property
    def dtype(self):
        return self.mem.dtype

    @property
    def sample_size(self):
        return self.size // self.shape[0]

    @property
    def matrix(self):
        return reshape(self.mem, (self.shape[0], self.sample_size))

    @property
    def plain(self):
        return ravel(self.mem)

    def init_unpickled(self):
        super(Vector, self).init_unpickled()
        self._unset_device()
        self.devmem = None
        self.map_arr_ = None
        self.map_flags = 0
        self.lock_ = threading.Lock()

    def min(self, *args, **kwargs):
        return self.mem.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self.mem.max(*args, **kwargs)

    def threadsafe(fn):
        def wrapped(self, *args, **kwargs):
            with self.lock_:
                res = fn(self, *args, **kwargs)
            return res
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_threadsafe'
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
        return self._mem.size if self._mem is not None else 0

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

    def _unset_device(self):
        def nothing(*args, **kwargs):
            pass

        for suffix in Vector.backend_methods:
            setattr(self, "_backend_" + suffix + "_", nothing)

    @threadsafe
    def initialize(self, device):
        if self._mem is None or self.devmem is not None or device is None:
            return

        self.device = device
        self._backend_realign_mem_()
        self._backend_create_devmem_()

        # Account mem in memwatcher
        if self.devmem is not None:
            global Watcher  # pylint: disable=W0601
            Watcher += self.devmem.size

    @threadsafe
    def map_read(self):
        return self._backend_map_read_()

    @threadsafe
    def map_write(self):
        return self._backend_map_write_()

    @threadsafe
    def map_invalidate(self):
        return self._backend_map_invalidate_()

    @threadsafe
    def unmap(self):
        return self._backend_unmap_()

    @threadsafe
    def reset(self, clear_hostmem=True):
        """Sets device buffers to None and optionally host buffer.
        """
        return self._reset(clear_hostmem)

    def _reset(self, clear_hostmem):
        self._backend_unmap_()
        if self.devmem is not None:
            global Watcher  # pylint: disable=W0601
            Watcher -= self.devmem.size
        self.devmem = None
        self.map_flags = 0
        if clear_hostmem:
            self._mem = None

    threadsafe = staticmethod(threadsafe)

    def ocl_create_devmem(self):
        self.devmem = self.device.queue_.context.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
            ravel(self._mem))

    def ocl_map(self, flags):
        if self.device is None:
            return
        if self.map_arr_ is not None:
            # already mapped properly, nothing to do
            if self.map_flags != cl.CL_MAP_READ or flags == cl.CL_MAP_READ:
                return
            self.ocl_unmap()
        if (flags == cl.CL_MAP_WRITE_INVALIDATE_REGION and
                self.device.device_info.version < 1.1999):
            # 'cause available only starting with 1.2
            flags = cl.CL_MAP_WRITE
        try:
            ev, self.map_arr_ = self.device.queue_.map_buffer(
                self.devmem, flags, self._mem.nbytes)
            del ev
        except cl.CLRuntimeError as err:
            self.error("Failed to map %d OpenCL bytes: %s(%d)",
                       self._mem.nbytes, str(err), err.code)
            raise
        if (int(cl.ffi.cast("size_t", self.map_arr_)) !=
                self._mem.__array_interface__["data"][0]):
            raise error.OpenCLError("map_buffer returned different pointer")
        self.map_flags = flags

    def ocl_map_read(self):
        self.ocl_map(cl.CL_MAP_READ)

    def ocl_map_write(self):
        self.ocl_map(cl.CL_MAP_WRITE)

    def ocl_map_invalidate(self):
        self.ocl_map(cl.CL_MAP_WRITE_INVALIDATE_REGION)

    def ocl_unmap(self):
        map_arr = self.map_arr_
        if map_arr is None:
            return
        # Workaround Python 3.4.0 incorrect destructor order call bug
        if self.device.queue_.handle is None:
            logging.getLogger("Vector").warning(
                "OpenCL device queue is None but Vector devmem was not "
                "explicitly unmapped.")
        elif self.devmem.handle is None:
            logging.getLogger("Vector").warning(
                "devmem.handle is None but Vector devmem was not "
                "explicitly unmapped.")
        else:
            self.device.queue_.unmap_buffer(self.devmem, map_arr,
                                            need_event=False)
        self.map_arr_ = None
        self.map_flags = 0

    def ocl_realign_mem(self):
        """We are using CL_MEM_USE_HOST_PTR, so memory should be PAGE-aligned.
        """
        if self.device is None or self.device.device_info.memalign <= 4096:
            memalign = 4096
        else:
            memalign = self.device.device_info.memalign
        self._mem = cl.realign_array(self._mem, memalign, numpy)

    def cuda_create_devmem(self):
        self.devmem = cu.MemAlloc(self.device.context, self.mem.nbytes)
        self.devmem.to_device(self.mem)

    def cuda_map_read(self):
        if self.device is None or self.map_flags >= 1:
            return
        self.devmem.to_host(self.mem)
        self.map_flags = 1

    def cuda_map_write(self):
        if self.device is None or self.map_flags >= 2:
            return
        if self.map_flags <= 1:
            self.devmem.to_host(self.mem)
        self.map_flags = 2

    def cuda_map_invalidate(self):
        if self.device is None or self.map_flags >= 1:
            return
        self.map_flags = 2

    def cuda_unmap(self):
        if self.map_flags <= 1:
            self.map_flags = 0
            return
        self.devmem.to_device_async(self.mem)
        self.map_flags = 0

    def cuda_realign_mem(self):
        """We are using simple cuMemAlloc, so host memory can be unaligned.
        """
        pass
