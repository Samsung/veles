# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 15, 2013

Data formats for connectors.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import cuda4py as cu
import numpy
import six
import os
import threading
import opencl4py as cl
if six.PY3:
    import atexit
    import weakref

from veles.backends import Device, NumpyDevice
from veles.compat import from_none
from veles.distributable import Pickleable
from veles.numpy_ext import (  # pylint: disable=W0611
    max_type, eq_addr, assert_addr, ravel, reshape, reshape_transposed,
    transpose, interleave, roundup, NumDiff)


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

    def __eq__(self, other):  # pylint: disable=C0203
        return self.mem == other.mem

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
        _map_arr_: address of the mapping if any exists.

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

    if six.PY3:
        __vectors__ = set()
        __registered = False

    def __init__(self, data=None):
        super(Vector, self).__init__()
        self._device = NumpyDevice()
        self.mem = data
        self._max_value = 1.0
        if six.PY3:
            # Workaround for unspecified destructor call order
            # This is a hard to reduce bug to report it
            Vector.__vectors__.add(weakref.ref(self))
            if not Vector.__registered:
                atexit.register(Vector.reset_all)
                Vector.__registered = True

    def __setstate__(self, state):
        super(Vector, self).__setstate__(state)
        if six.PY3:
            Vector.__vectors__.add(weakref.ref(self))

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device is None:
            raise ValueError("device may not be None")
        if not isinstance(device, Device):
            raise TypeError(
                "device must be an instance of veles.backends.Device, got %s" %
                device)
        if isinstance(device, NumpyDevice):
            self._reset(self.mem)
            self._unset_device()
            return
        self._reset(self.mem)
        self._device = device
        for suffix in Vector.backend_methods:
            setattr(self, "_backend_" + suffix + "_",
                    getattr(self, device.backend_name + "_" + suffix))

    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, value):
        if self.devmem is not None and not eq_addr(self.mem, value):
            raise ValueError(
                "Device buffer has already been assigned, call reset() "
                "beforehand.")
        if value is not None and not isinstance(value, numpy.ndarray):
            raise TypeError(
                "Attempted to set Vector's mem to something which is not a "
                "numpy array: %s of type %s" % (value, type(value)))
        self._mem = value

    @property
    def devmem(self):
        return self._devmem_

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

    @shape.setter
    def shape(self, value):
        self.mem.shape = value

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
        self._devmem_ = None
        self._map_arr_ = None
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
        if self.device is None:
            import gc

            for key, val in gc.get_referrers(self)[0].items():
                if val is self:
                    print(key)
        if self.device.pid == os.getpid():
            self.map_read()
        return super(Vector, self).__getstate__()

    def __bool__(self):
        return self._mem is not None and len(self._mem) > 0

    def __nonzero__(self):
        return self.__bool__()

    def __lshift__(self, value):
        self.mem = value

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

    if six.PY3:
        @staticmethod
        def reset_all():
            for ref in Vector.__vectors__:
                vec = ref()
                if vec is not None:
                    vec.reset()

    def _unset_device(self):
        def nothing(*args, **kwargs):
            pass

        for suffix in Vector.backend_methods:
            setattr(self, "_backend_" + suffix + "_", nothing)

    @threadsafe
    def initialize(self, device):
        if device is None:
            raise ValueError("device may not be None")
        if self.device == device and self.devmem is not None:
            # Check against double initialization (pretty legal)
            return
        if self.mem is not None or isinstance(device, NumpyDevice):
            # Set the device only if it makes sense
            self.device = device
        if self.mem is None or isinstance(self.device, NumpyDevice):
            # We are done if there is no host buffer or device
            return

        assert isinstance(self.mem, numpy.ndarray), \
            "Wrong mem type: %s" % type(self.mem)
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
    def reset(self, new_mem=None):
        """Sets device buffers to None and optionally release the host buffer.
        :param new_mem: Set "mem" property to this value.
        """
        return self._reset(new_mem)

    def _reset(self, new_mem):
        """
        :param new_mem: mem will be set to this value. Can be None.
        :return: Nothing.
        """
        self._backend_unmap_()
        if self.devmem is not None:
            global Watcher  # pylint: disable=W0601
            Watcher -= self.devmem.size
        self._devmem_ = None
        self.map_flags = 0
        self.mem = new_mem

    threadsafe = staticmethod(threadsafe)

    def ocl_create_devmem(self):
        self._devmem_ = self.device.queue_.context.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, self.plain)

    def ocl_map(self, flags):
        if isinstance(self.device, NumpyDevice):
            return
        if self._map_arr_ is not None:
            # already mapped properly, nothing to do
            if self.map_flags != cl.CL_MAP_READ or flags == cl.CL_MAP_READ:
                return
            self.ocl_unmap()
        if (flags == cl.CL_MAP_WRITE_INVALIDATE_REGION and
                self.device.device_info.version < 1.1999):
            # 'cause available only starting with 1.2
            flags = cl.CL_MAP_WRITE
        assert self.devmem is not None
        try:
            ev, self._map_arr_ = self.device.queue_.map_buffer(
                self.devmem, flags, self._mem.nbytes)
            del ev
        except cl.CLRuntimeError as err:
            self.error("Failed to map %d OpenCL bytes: %s(%d)",
                       self._mem.nbytes, str(err), err.code)
            raise
        if (int(cl.ffi.cast("size_t", self._map_arr_)) !=
                self._mem.__array_interface__["data"][0]):
            raise RuntimeError("map_buffer returned different pointer")
        self.map_flags = flags

    def ocl_map_read(self):
        self.ocl_map(cl.CL_MAP_READ)

    def ocl_map_write(self):
        self.ocl_map(cl.CL_MAP_WRITE)

    def ocl_map_invalidate(self):
        self.ocl_map(cl.CL_MAP_WRITE_INVALIDATE_REGION)

    def ocl_unmap(self):
        map_arr = self._map_arr_
        if map_arr is None:
            return
        # The following checks are neccessary in case of cyclic object
        # references and CPython >= 3.4.0
        if self.device.queue_.handle is None:
            self.warning(
                "%s: OpenCL device queue is None but Vector devmem was not "
                "explicitly unmapped.", self)
        elif self.devmem.handle is None:
            self.warning(
                "%s: devmem.handle is None but Vector devmem was not "
                "explicitly unmapped.", self)
        else:
            self.device.queue_.unmap_buffer(self.devmem, map_arr,
                                            need_event=False)
        self._map_arr_ = None
        self.map_flags = 0

    def ocl_realign_mem(self):
        """We are using CL_MEM_USE_HOST_PTR, so memory should be PAGE-aligned.
        """
        if isinstance(self.device, NumpyDevice) or \
                self.device.device_info.memalign <= 4096:
            memalign = 4096
        else:
            memalign = self.device.device_info.memalign
        self.mem = cl.realign_array(self._mem, memalign, numpy)

    def cuda_create_devmem(self):
        self._devmem_ = cu.MemAlloc(self.device.context, self.plain.nbytes)
        self.devmem.to_device(self.mem)

    def cuda_map_read(self):
        if isinstance(self.device, NumpyDevice) or self.map_flags >= 1:
            return
        self.devmem.to_host(self.mem)
        self.map_flags = 1

    def cuda_map_write(self):
        if isinstance(self.device, NumpyDevice) or self.map_flags >= 2:
            return
        if self.map_flags < 1:  # there were no map_read before
            self.devmem.to_host(self.mem)  # sync copy
        self.map_flags = 2

    def cuda_map_invalidate(self):
        if isinstance(self.device, NumpyDevice) or self.map_flags >= 2:
            return
        if self.map_flags < 1:  # there were no map_read before
            self.device.sync()  # sync without copy
        self.map_flags = 2

    def cuda_unmap(self):
        if self.map_flags <= 1:
            self.map_flags = 0
            return
        self.devmem.to_device_async(self.mem)
        self.map_flags = 0

    def cuda_realign_mem(self):
        # We expect numpy array with continuous memory layout, so realign it.
        # PAGE-boundary alignment may increase speed also.
        self.mem = cl.realign_array(self._mem, 4096, numpy)
