# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 21, 2013

OpenCL base classes.

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

import argparse
import cuda4py as cu
import cuda4py.blas as cublas
import gc
from importlib import import_module
import json
import numpy
import opencl4py as cl
import os
from psutil import virtual_memory
from six import add_metaclass
import sys
from threading import current_thread

from veles.cmdline import CommandLineArgumentsRegistry
from veles.compat import from_none
from veles.config import root
from veles.distributable import Pickleable
from veles.logger import Logger
import veles.opencl_types as opencl_types
import veles.external.prettytable as prettytable


PYVER = sys.version_info[0]


class DeviceInfo(Logger):
    """Info about device.

    Attributes:
        desc: Description of the device.
        memsize: "available" size of the memory on the device.
        memalign: best alignment for device buffers.
        version: OpenCL version.
        rating: in [0, 1] interval (1 - fastest, 0.5 - 50% slower than fastest,
                0 - unrated).
        device_info: contains block sizes for different kernel types.
    """
    def __init__(self, **kwargs):
        super(DeviceInfo, self).__init__()
        self.desc = kwargs["desc"]
        self.memsize = kwargs["memsize"]
        self.memalign = kwargs["memalign"]
        self.version = kwargs["version"]
        self.device_type = kwargs["device_type"]
        self.max_work_group_size = kwargs["max_work_group_size"]
        self.max_work_item_sizes = kwargs["max_work_item_sizes"]
        self.local_memsize = kwargs["local_memsize"]
        self.rating = {}
        self.device_info = {}

    def get_kernel_bs_vo(self, **kwargs):
        """Gets optimal block size and vector_opt
        flag for matrix multiplication.

        Parameters:
            dtype: numeric data type as string (float or double).
            kernel: hint for the name of the kernel for which the optimal
                    block sizes will be returned:
                    conv: convolutional forward propagation,
                    deconv: convolutional back propagation,
                    all other: simple matrix multiplication.
            precision: precision level for summation (0, 1, 2)
                       (defaults to root.common.engine.precision_level).

        Returns:
            BLOCK_SIZE, VECTOR_OPT
        """
        dtype = kwargs["dtype"]
        if not isinstance(dtype, str):
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        krnnme = kwargs.get("kernel", "matrix_multiplication")
        precision = kwargs.get("precision", root.common.engine.precision_level)
        krninfo = self.device_info.get(krnnme)
        if krninfo is None:
            # Benchmark for other kernel types is not implemented,
            # so only debug level here
            self.debug(
                "Kernel \"%s\" was not found, "
                "rolling back to block size for matrix_multiplication",
                krnnme)
            krnnme = "matrix_multiplication"
            krninfo = self.device_info.get(krnnme)
            if krninfo is None:
                bs = 8
                self.warning(
                    "krnnme = %s was not found, "
                    "will use block size %d", krnnme, bs)
                return bs, False
        typeinfo = krninfo.get(dtype)
        if typeinfo is None:
            bs = 8
            self.warning(
                "dtype = %s was not found with krnnme = %s, "
                "will use block size %d", dtype, krnnme, bs)
            return bs, False
        bs_dt = typeinfo.get(str(precision))
        while bs_dt is None and precision > 0:
            precision -= 1
            bs_dt = typeinfo.get(str(precision))
        if bs_dt is None:
            bs = 8
            self.warning(
                "precision = 0 was not found with krnnme = %s and dtype = %s, "
                "will use block size %d", krnnme, dtype, bs)
            return bs, False
        return bs_dt[0], bs_dt[1]

    def get_max_block_size(self, dtype, vector_opt):
        itemsize = {"float": 4, "double": 8}[dtype]
        sz = int(numpy.sqrt(self.max_work_group_size))
        sh = self.max_work_item_sizes
        bs = min(sz, sh[0], sh[1])
        while bs * bs * 2 * itemsize > self.local_memsize:
            bs -= 1
        if vector_opt:  # round down to 4
            bs >>= 2
            bs <<= 2
        return bs

    @property
    def is_cpu(self):
        return self.device_type == cl.CL_DEVICE_TYPE_CPU


class DeviceNotFoundError(Exception):
    pass


class BackendRegistry(CommandLineArgumentsRegistry):
    backends = {}

    def __init__(cls, name, bases, clsdict):
        super(BackendRegistry, cls).__init__(name, bases, clsdict)
        try:
            BackendRegistry.backends[clsdict["BACKEND"]] = cls
        except KeyError:
            raise from_none(KeyError("%s does not define BACKEND" % cls))
        assert "PRIORITY" in clsdict, "%s does not define PRIORITY" % cls

    @staticmethod
    def backends_as_str():
        return ", ".join("\"%s\" for %s" % (k, v.__name__) for k, v in sorted(
            BackendRegistry.backends.items()))


@add_metaclass(CommandLineArgumentsRegistry)
class Device(Pickleable):
    """Base device class.

    Attributes:
        _pid_: process id.
    """
    def __new__(cls, *args, **kwargs):
        assert issubclass(cls, Device)
        backend = kwargs.get(
            "backend", os.getenv("VELES_BACKEND", root.common.engine.backend))
        cls = BackendRegistry.backends[backend]
        if cls.__new__ != Device.__new__:
            return cls.__new__(cls, *args, **kwargs)
        return object.__new__(cls)

    def init_unpickled(self):
        super(Device, self).init_unpickled()
        self._pid_ = os.getpid()
        self._thread_pool_detach_callbacks_ = {}
        self._temp_buffer_size = 0
        self._temp_buffer_ = None

    def __del__(self):
        for pool in dict(self._thread_pool_detach_callbacks_):
            self.thread_pool_detach(pool)

    @property
    def backend_name(self):
        """Returns name of the backend.
        """
        return type(self).BACKEND

    @property
    def id(self):
        return None

    @property
    def pid(self):
        """Process ID.
        """
        return self._pid_

    @property
    def blas(self):
        """Returns BLAS instance.
        """
        return None

    @property
    def is_async(self):
        return type(self).ASYNC

    def sync(self):
        """Synchronizes the device execution queue.
        """
        pass

    def is_attached(self, thread_pool):
        return thread_pool in self._thread_pool_detach_callbacks_

    def assign_backend_methods(self, obj, backend_methods):
        """Scans backends class hierarchy and assigns found methods to obj.
        """
        for suffix in backend_methods:
            # Scan class hierarchy for backend method
            checked = []  # this is just for exception message
            for cls in type(self).mro():
                backend = getattr(cls, "BACKEND", None)
                if backend is None:
                    continue
                checked.append(cls)
                backend_method = getattr(obj, backend + "_" + suffix, None)
                if backend_method is not None:
                    break
            else:
                raise AttributeError(
                    "No implementation of %s with backends %s found in %s " %
                    (suffix, checked, type(obj)))
            setattr(obj, "_backend_" + suffix + "_", backend_method)

    def thread_pool_attach(self, thread_pool):
        if thread_pool in self._thread_pool_detach_callbacks_:
            self.warning("Already is_attached to %s", thread_pool)
            return
        self._register_thread_pool_callbacks(thread_pool)

        def detach():
            self.thread_pool_detach(thread_pool)

        self._thread_pool_detach_callbacks_[thread_pool] = detach
        thread_pool.register_on_shutdown(detach)

    def thread_pool_detach(self, thread_pool):
        if thread_pool not in self._thread_pool_detach_callbacks_:
            self.warning("Unable to detach from %s: not attached", thread_pool)
            return
        thread_pool.unregister_on_shutdown(
            self._thread_pool_detach_callbacks_[thread_pool])
        del self._thread_pool_detach_callbacks_[thread_pool]
        self._unregister_thread_pool_callbacks(thread_pool)

    def _register_thread_pool_callbacks(self, pool):
        """Registers callbacks for the thread pool.
        """
        # Important! Save the bound method to variable to avoid dead weak refs
        # See http://stackoverflow.com/questions/19443440/weak-reference-to-python-class-method  # nopep8
        self._on_thread_enter_ = self._on_thread_enter
        self._on_thread_exit_ = self._on_thread_exit
        pool.register_on_thread_enter(self._on_thread_enter_)
        pool.register_on_thread_exit(self._on_thread_exit_)

    def _unregister_thread_pool_callbacks(self, pool):
        pool.unregister_on_thread_enter(self._on_thread_enter)
        pool.unregister_on_thread_exit(self._on_thread_exit)

    @staticmethod
    def iterparse(text):
        bpos = text.find(':')
        if bpos < 1:
            raise ValueError("Invalid devices definition: %s" % text)
        backend, devs = text[:bpos], text[bpos + 1:]
        multidevs = devs.split('x')
        for _ in range(1 if len(multidevs) == 1 else int(multidevs[1])):
            for d in BackendRegistry.backends[backend].iterparse(multidevs[0]):
                yield d

    def _on_thread_enter(self):
        """Called justed after the new thread has been created
        in the thread pool.
        """
        pass

    def _on_thread_exit(self):
        """Called just before the thread will be terminated.
        """
        pass

    @property
    def exists(self):
        """Returns True if device is ready for use.
        """
        return False

    @staticmethod
    def arg_completer(prefix, **kwargs):
        def format_device(plf, dev):
            return "%s - %s on %s" % (dev.path, dev.name.strip(), plf.name)

        if prefix.strip() == "":
            platforms = cl.Platforms().platforms
            if len(platforms) == 1 and len(platforms[0].devices) == 1:
                return ["0:0"]
            result = []
            for platform in platforms:
                for device in platform:
                    result.append(format_device(platform, device))
            return result
        parsed = [p for p in prefix.split(':') if p.strip() != ""]
        platform = cl.Platforms().platforms[int(parsed[0].strip())]
        if len(parsed) == 1:
            if len(platform.devices) == 1:
                return [platform.devices[0].path]
            result = []
            for device in platform:
                result.append(format_device(platform, device))
            return result

    @staticmethod
    def init_parser(**kwargs):
        parser = kwargs.get("parser", argparse.ArgumentParser())

        def set_backend(name):
            if name not in BackendRegistry.backends:
                raise ValueError(
                    "Insupported backend name: %s. Choose any from %s." %
                    (name, BackendRegistry.backends_as_str()))
            root.common.engine.backend = name

        parser.add_argument(
            "-d", "--device", type=str, default="",
            help="Device ID to use. E.g. 0:1 for OpenCL or 1 for CUDA.") \
            .completer = Device.arg_completer
        parser.add_argument(
            "-a", "--backend", type=set_backend, default="auto",
            help="Acceleration backend to use. Currently supported values are "
                 "%s." % BackendRegistry.backends_as_str())
        return parser

    @staticmethod
    def parse_device(**kwargs):
        parser = Device.init_parser(**kwargs)
        args, _ = parser.parse_known_args(Device.class_argv)
        return args.device

    def _alloc_temp_buffer(self, size):
        """Allocates device-specific temporary buffer.
        """
        raise NotImplementedError()

    def request_temp_buffer(self, size):
        """Tries to allocate the device specific temporary buffer
        of max(size, previous max size of this function call) bytes,
        then frees it.
        """
        if size & 1:
            size += 1
        if size > self._temp_buffer_size:
            # Try to allocate the memory on the device
            buf = self._alloc_temp_buffer(size)
            del buf
            self._temp_buffer_size = size

    def get_temp_buffer(self):
        """Returns allocated device-specific temporary buffer.
        """
        if self._temp_buffer_ is None:
            self._temp_buffer_ = self._alloc_temp_buffer(
                self._temp_buffer_size)
        return self._temp_buffer_


@add_metaclass(BackendRegistry)
class AutoDevice(Device):
    """
    Overrides __new__() to automatically select the best available Device type.
    """
    BACKEND = "auto"
    PRIORITY = 0

    def __new__(cls, *args, **kwargs):
        for cls in sorted(BackendRegistry.backends.values(),
                          key=lambda b: b.PRIORITY, reverse=True):
            if cls.available():
                return object.__new__(cls, *args)
        assert False, "Impossible because numpy backend is always available"

    @staticmethod
    def available():
        return False


@add_metaclass(BackendRegistry)
class OpenCLDevice(Device):
    """OpenCL device class.

    Attributes:
        device_info: DeviceInfo object.
        context_: OpenCL context handle.
        queue_: OpenCL device queue.
        pid_: process id.
    """

    BACKEND = "ocl"
    PRIORITY = 20
    DEVICE_INFOS_JSON = "device_infos.json"
    ASYNC = True
    skip = cl.skip

    # Allow this class to be created manually
    def __new__(cls, *args):
        return object.__new__(cls, *args)

    def __init__(self):
        super(OpenCLDevice, self).__init__()

        self._blas = None
        self._id = None

        # Workaround for NVIDIA
        # (fixes incorrect behaviour with OpenCL binaries)
        if os.getenv("CUDA_CACHE_DISABLE") is None:
            os.putenv("CUDA_CACHE_DISABLE", "1")

        # Workaround for AMD
        # (fixes segmentation fault when accessed over ssh with X and
        #  no X is running or when accessing locally and integrated
        #  video device is used instead of AMD one)
        d = os.getenv("DISPLAY")
        if d is not None and d != os.getenv("COMPUTE"):
            os.unsetenv("DISPLAY")

        # Set 64-bit mode for AMD OpenCL by default
        if os.getenv("GPU_FORCE_64BIT_PTR") is None:
            os.putenv("GPU_FORCE_64BIT_PTR", "1")

        # Get the device
        res = self._get_some_device()

        # Restore DISPLAY to enable drawing
        if d is not None:
            os.putenv("DISPLAY", d)
        if not res:
            return

        self._fill_device_info_performance_values()
        log_configs = "Selected the following OpenCL configuration:\n"
        table = prettytable.PrettyTable("device", " dtype", "rating",
                                        "BLOCK_SIZE", "VECTOR_OPT", "version")
        table.align["device"] = "l"
        table.align[" dtype"] = "l"
        table.align["BLOCK_SIZE"] = "l"
        table.align["VECTOR_OPT"] = "l"
        for dtype in sorted(opencl_types.dtypes.keys()):
            rating = self.device_info.rating.get(dtype)
            if rating is None:
                rating = ""
            else:
                rating = "%.3f" % rating
            bs_vo = self.device_info.get_kernel_bs_vo(dtype=dtype)
            table.add_row(self.device_info.desc, dtype, rating,
                          bs_vo[0], bs_vo[1], self.device_info.version)
        self.info(log_configs + str(table))

    @property
    def id(self):
        return self._id

    @property
    def blas(self):
        return self._blas

    @blas.setter
    def blas(self, value):
        assert self._blas is None, "blas property is not None already"
        self._blas = value

    @property
    def exists(self):
        return self.queue_ is not None

    def init_unpickled(self):
        super(OpenCLDevice, self).init_unpickled()
        self.queue_ = None

    @property
    def max_group_size(self):
        return self.queue_.device.max_work_group_size

    @staticmethod
    def iterparse(text):
        oclpnums, ocldevnum = text.split(':')
        oclpnum = int(oclpnums)
        ocldevarr = ocldevnum.split('-')
        ocldevmin = int(ocldevarr[0])
        if len(ocldevarr) > 1:
            ocldevmax = int(ocldevarr[1])
        else:
            ocldevmax = ocldevmin
        for dev in range(ocldevmin, ocldevmax + 1):
            yield OpenCLDevice.BACKEND, "%d:%d" % (oclpnum, dev)

    def assign_backend_methods(self, obj, backend_methods):
        super(OpenCLDevice, self).assign_backend_methods(obj, backend_methods)
        obj.skip_args = cl.skip

    def compute_ratings(self, device_infos):
        devdt = {}
        min_dt = {}
        for desc, device_info in sorted(device_infos.items()):
            krninfo = device_info.get("matrix_multiplication")
            if krninfo is None:
                continue
            devdt[desc] = {}
            for dtype, typeinfo in krninfo.items():
                bs_vo_dt = typeinfo.get("0")
                if bs_vo_dt is None or len(bs_vo_dt) < 3:
                    continue
                devdt[desc][dtype] = bs_vo_dt[2]
                min_dt[dtype] = min(min_dt.get(dtype, 1.0e30), bs_vo_dt[2])

        table = prettytable.PrettyTable("device", " dtype", "rating")
        table.align["device"] = "l"
        table.align[" dtype"] = "l"
        rating = {}
        for desc, dtypedt in sorted(devdt.items()):
            rating[desc] = {}
            for dtype, dt in sorted(dtypedt.items()):
                rating[desc][dtype] = min_dt[dtype] / dt
                table.add_row(desc, dtype, "%.3f" % rating[desc][dtype])
        self.debug("Device ratings:\n%s", str(table))

        if self.device_info.desc in rating:
            self.device_info.rating = rating[self.device_info.desc]

    def sync(self):
        self.queue_.flush()
        self.queue_.finish()

    @staticmethod
    def available():
        try:
            return len(cl.Platforms().platforms) > 0
        except:
            return False

    def _get_some_device(self, **kwargs):
        """Gets some device from the available OpenCL devices.
        Returns True if any device was selected, otherwise, False.
        """
        device = self.parse_device(**kwargs)
        try:
            platforms = cl.Platforms()
        except cl.CLRuntimeError:
            platforms = None
        if platforms is None or len(platforms.platforms) == 0:
            raise DeviceNotFoundError("No OpenCL devices were found")
        self._id = device
        if device == "":
            context = platforms.create_some_context()
        else:
            platfnum, devnums = device.split(':')
            try:
                platform = platforms.platforms[int(platfnum)]
            except IndexError:
                raise from_none(
                    DeviceNotFoundError("Device %s was not found." % device))
            context = platform.create_context(
                [platform.devices[int(devnum)]
                 for devnum in devnums.split(',')])
        if "NVIDIA" in context.platform.name:
            def fail(*args, **kwargs):
                raise RuntimeError("fork() breaks NVIDIA OpenCL")

            os.fork = fail
            import subprocess

            subprocess.Popen = fail
        device = context.devices[0]
        desc = "%s/%s" % (device.vendor.strip(), device.name.strip())
        self.queue_ = context.create_queue(device)
        self.device_info = DeviceInfo(
            desc=desc, memsize=device.memsize,
            memalign=device.memalign, version=device.version,
            device_type=device.type,
            max_work_group_size=self.queue_.device.max_work_group_size,
            max_work_item_sizes=self.queue_.device.max_work_item_sizes,
            local_memsize=self.queue_.device.local_memsize)
        return True

    def _fill_device_info_performance_values(self):
        device_infos = {}
        found_any = False
        for devdir in root.common.engine.device_dirs:
            try:
                os.makedirs(devdir, 0o755)
            except OSError:
                pass
            device_infos_fnme = os.path.join(devdir,
                                             OpenCLDevice.DEVICE_INFOS_JSON)
            try:
                with open(device_infos_fnme, "r") as fin:
                    device_infos.update(json.load(fin))
                found_any = True
            except IOError:
                pass
            except ValueError as e:
                self.warning("Failed to load %s: %s", device_infos_fnme, e)
        if not found_any:
            self.warning("Did not find %s in any of the configured paths: %s",
                         OpenCLDevice.DEVICE_INFOS_JSON,
                         root.common.engine.device_dirs)
        if ((self.device_info.desc not in device_infos and
                root.common.engine.test_unknown_device) or
                (self.device_info.desc in device_infos and
                 root.common.engine.test_known_device)):
            self.warning("%s, will perform a "
                         "quick test now.", "Forced device retest"
                         if self.device_info.desc in device_infos
                         else "Device has not been analyzed yet")
            self._find_optimal_bs_vo(device_infos)
            found_any = False
            for devdir in root.common.engine.device_dirs:
                device_infos_fnme = os.path.join(
                    devdir, OpenCLDevice.DEVICE_INFOS_JSON)
                try:
                    with open(device_infos_fnme, "w") as fout:
                        json.dump(device_infos, fout, indent=2, sort_keys=True)
                    found_any = True
                except IOError:
                    pass
            if not found_any:
                self.warning("Unable to save the analysis results to any of "
                             "the configured paths: %s",
                             root.common.engine.device_dirs)
        self.compute_ratings(device_infos)
        if self.device_info.desc in device_infos:
            self.device_info.device_info = device_infos[self.device_info.desc]

    def _find_optimal_bs_vo(self, device_infos):
        device_info = device_infos.get(self.device_info.desc, {})
        krnnme = "matrix_multiplication"
        if krnnme not in device_info:
            device_info[krnnme] = {}
        # FIXME(v.markovtsev): disable R0401 locally when pylint issue is fixed
        # https://bitbucket.org/logilab/pylint/issue/61
        # pylint: disable=R0401
        dummy = import_module("veles.dummy")
        opencl_units = import_module("veles.accelerated_units")
        benchmark = opencl_units.DeviceBenchmark
        for dtype in root.common.engine.test_precision_types:
            if dtype not in device_info[krnnme]:
                device_info[krnnme][dtype] = {}
            # json wants strings
            for precision_level in (
                    str(p) for p in root.common.engine.test_precision_levels):
                min_dt = 1.0e30
                for vector_opt in (False, True):
                    max_block_size = self.device_info.get_max_block_size(
                        dtype, vector_opt)
                    min_block_size = 8
                    if int(vector_opt):
                        min_block_size >>= 2
                        min_block_size <<= 2
                        bs_inc = 4
                    else:
                        bs_inc = 1
                    for block_size in range(min_block_size, max_block_size + 1,
                                            bs_inc):
                        self.info(
                            "Testing %s dtype=%s precision_level=%s "
                            "block_size=%d vector_opt=%s",
                            krnnme, dtype, precision_level,
                            block_size, vector_opt)
                        try:
                            with dummy.DummyWorkflow() as wf:
                                u = benchmark(
                                    wf, size=3001, repeats=3,
                                    dtype=dtype,
                                    precision_level=precision_level,
                                    block_size=block_size,
                                    vector_opt=vector_opt,
                                    return_time=True, dry_run_first=True)
                                u.initialize(self)
                                dt = u.run()
                        except cl.CLRuntimeError as e:
                            self.exception("Failed to evaluate block size %d",
                                           block_size)
                            if e.code == -5:  # CL_OUT_OF_RESOURCES
                                break
                            else:
                                continue
                        finally:
                            gc.collect()
                        if dt < min_dt:
                            min_dt = dt
                            device_info[krnnme][dtype][precision_level] = (
                                block_size, bool(int(vector_opt)), min_dt)
        device_infos[self.device_info.desc] = device_info

    def _alloc_temp_buffer(self, size):
        # Allocate the buffer
        buf = self.queue_.context.create_buffer(
            cl.CL_MEM_READ_WRITE, size=size)
        self.sync()
        # Move it to device
        self.queue_.copy_buffer(buf, buf, 0, size >> 1, size >> 1)
        self.sync()
        return buf


@add_metaclass(BackendRegistry)
class CUDADevice(Device):
    """CUDA device class.

    Attributes:
        _context_: CUDA context handle.
        _blas_: dictionary of thread-id => CUBLAS instances.
    """

    BACKEND = "cuda"
    PRIORITY = 30
    ASYNC = True
    skip = cu.skip

    # Allow this class to be created manually
    def __new__(cls, *args):
        return object.__new__(cls, *args)

    def __init__(self):
        super(CUDADevice, self).__init__()
        self._context_ = None
        self._id = None
        self._blas_ = {}

        # Get the device
        self._get_some_device()

        log_configs = "Selected the following CUDA device:\n"
        table = prettytable.PrettyTable("device", "mem", "compute", "pci")
        table.align["device"] = "l"
        table.align["mem"] = "r"
        table.align["pci"] = "l"
        table.add_row(
            self.context.device.name, self.context.device.total_mem // 1048576,
            "%d.%d" % self.context.device.compute_capability,
            self.context.device.pci_bus_id)
        self.info(log_configs + str(table))

    @property
    def id(self):
        return self._id

    @property
    def context(self):
        return self._context_

    @property
    def exists(self):
        return self._context_ is not None

    def suggest_block_size(self, krn):
        if krn is None:
            raise ValueError("Received None as an argument")
        _min_grid_size, block_size = krn.max_potential_block_size()
        ab_best = krn.max_active_blocks_per_multiprocessor(block_size)
        ab = ab_best
        # Warp size usually 32 and no improvements over 128 with NVIDIA Titan
        min_size = self.context.device.warp_size * 4
        best_block_size = None
        while (ab >= ab_best and not (block_size & 1) and
                block_size >= min_size):
            ab_best = ab
            best_block_size = block_size
            block_size >>= 1
            ab = krn.max_active_blocks_per_multiprocessor(block_size)
        return best_block_size

    def _register_thread_pool_callbacks(self, pool):
        super(CUDADevice, self)._register_thread_pool_callbacks(pool)
        self.context.push_current()

    def _unregister_thread_pool_callbacks(self, pool):
        super(CUDADevice, self)._unregister_thread_pool_callbacks(pool)
        self.context.pop_current()

    def _on_thread_enter(self):
        self._context_.push_current()

    def _on_thread_exit(self):
        tid = current_thread().ident
        if tid in self._blas_:
            blas = self._blas_.pop(tid)
            del blas
        self._context_.pop_current()

    @property
    def blas(self):
        tid = current_thread().ident
        blas = self._blas_.get(tid)
        if blas is None:
            blas = cublas.CUBLAS(self.context)
            self._blas_[tid] = blas
        return blas

    @staticmethod
    def arg_completer(prefix, **kwargs):
        def format_device(dev):
            return "%d: %s - %s, %dMb, compute_%d%d, pci %s" % ((
                dev.handle, dev.name, dev.total_mem) +
                dev.compute_capability + (dev.pci_bus_id,))

        devices = cu.Devices()
        if len(devices) == 1:
            return ["0"]
        result = []
        for device in devices:
            result.append(format_device(device))
        return result

    @staticmethod
    def iterparse(text):
        ocldevarr = text.split('-')
        ocldevmin = int(ocldevarr[0])
        if len(ocldevarr) > 1:
            ocldevmax = int(ocldevarr[1])
        else:
            ocldevmax = ocldevmin
        for dev in range(ocldevmin, ocldevmax + 1):
            yield CUDADevice.BACKEND, str(dev)

    def assign_backend_methods(self, obj, backend_methods):
        super(CUDADevice, self).assign_backend_methods(obj, backend_methods)
        obj.skip_args = cu.skip

    def _get_some_device(self, **kwargs):
        """Gets some device from the available CUDA devices.
        Returns True if any device was selected, otherwise, False.
        """
        device = self.parse_device(**kwargs)
        try:
            devices = cu.Devices()
        except (OSError, cu.CUDARuntimeError):
            devices = None
        if devices is None or not len(devices):
            raise DeviceNotFoundError("No CUDA devices were found")
        self._id = device
        if device == "":
            context = devices.create_some_context()
        else:
            try:
                device = devices[int(device)]
            except IndexError:
                raise from_none(
                    DeviceNotFoundError(
                        "CUDA device %s was not found." % device))
            context = device.create_context()
        self._context_ = context

        device = self.context.device
        self.device_info = DeviceInfo(
            desc=device.name, memsize=device.total_mem,
            memalign=4096, version=device.compute_capability,
            device_type="CUDA",
            max_work_group_size=device.max_grid_dims,
            max_work_item_sizes=device.max_block_dims,
            local_memsize=device.max_shared_memory_per_block)
        return True

    def sync(self):
        self.context.synchronize()

    @staticmethod
    def available():
        try:
            return len(cu.Devices()) > 0
        except:
            return False

    def _alloc_temp_buffer(self, size):
        # Allocate the buffer
        return cu.MemAlloc(self.context, size)


@add_metaclass(BackendRegistry)
class NumpyDevice(Device):
    """Python numpy pseudo device class.
    """

    BACKEND = "numpy"
    PRIORITY = 10
    ASYNC = False

    def __new__(cls, *args):
        return object.__new__(cls)

    def __init__(self):
        super(NumpyDevice, self).__init__()
        self.device_info = DeviceInfo(
            desc="Python Numpy", memsize=virtual_memory().total,
            memalign=8, version=numpy.__version__,
            device_type="Hybrid",
            max_work_group_size=None, max_work_item_sizes=None,
            local_memsize=virtual_memory().total)

    @staticmethod
    def available():
        return True

    @staticmethod
    def iterparse(text):
        yield "numpy", "numpy"

    def _alloc_temp_buffer(self, size):
        # Allocate the buffer
        return numpy.zeros(size, dtype=numpy.uint8)
