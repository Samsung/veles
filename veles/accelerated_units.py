# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 25, 2014

Units that use hardware acceleration, either OpenCL, CUDA or other supported
backend.

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
from copy import copy
from jinja2 import Template, TemplateError
import logging
try:
    from numba import jit
except (ImportError, AttributeError):
    jit = None
import numpy
import os
import re
from six import BytesIO, add_metaclass, PY3
import tarfile
from tempfile import NamedTemporaryFile
import time
from zope.interface import implementer, Interface

from veles.compat import from_none
from veles.config import root
from veles.memory import Array, roundup
import veles.opencl_types as opencl_types
from veles.backends import Device, OpenCLDevice, CUDADevice, NumpyDevice
from veles.pickle2 import pickle, best_protocol
from veles.timeit2 import timeit
from veles.units import Unit, IUnit, UnitCommandLineArgumentsRegistry
from veles.workflow import Workflow


class IncludeError(Exception):
    pass


class INumpyUnit(Interface):
    def numpy_init():
        """
        Initialize Numpy-specific stuff. Normally, this is a no-op, so
        AcceleratedUnit (base class) provides it out of the box.
        """

    def numpy_run():
        """
        Run using Numpy functions.
        """


class IOpenCLUnit(Interface):
    """Requires cpu and ocl methods for OpenCLUnit descendants.
    """

    def ocl_init():
        """
        Initialize OpenCL-specific stuff. Called inside initialize().
        """

    def ocl_run():
        """Run on GPU/any OpenCL capable device.
        """

    def initialize(device, **kwargs):
        """initialize() with "device" obligatory argument.
        """


class ICUDAUnit(Interface):
    """Requires cpu and cuda methods for CUDAUnit descendants.
    """

    def cuda_init():
        """
        Initialize CUDA-specific stuff. Called inside initialize().
        """

    def cuda_run():
        """Run on GPU/any CUDA capable device.
        """

    def initialize(device, **kwargs):
        """initialize() with "device" obligatory argument.
        """


INTERFACE_MAPPING = {OpenCLDevice: IOpenCLUnit, CUDADevice: ICUDAUnit,
                     NumpyDevice: INumpyUnit}


@implementer(IUnit)
@add_metaclass(UnitCommandLineArgumentsRegistry)
class AcceleratedUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        program_: OpenCL program.
        sources_: OpenCL source files: file => defines.
        cache: whether to cache the compiled OpenCL/CUDA programs.
    """
    backend_methods = ("run", "init", "build_program", "get_kernel",
                       "execute_kernel")
    hide_from_registry = True

    def __init__(self, workflow, **kwargs):
        super(AcceleratedUnit, self).__init__(workflow, **kwargs)
        self._device = NumpyDevice()
        self._cache = kwargs.get("cache", True)
        # Yup, this is right - self._force_numpy is initialized in
        # init_unpickled
        self._force_numpy = kwargs.get("force_numpy", self._force_numpy)
        self.intel_opencl_workaround = \
            root.common.engine.force_numpy_run_on_intel_opencl

    def init_unpickled(self):
        super(AcceleratedUnit, self).init_unpickled()
        self.program_ = None
        self.sources_ = {}
        parser = AcceleratedUnit.init_parser()
        args, _ = parser.parse_known_args(self.argv)
        if not hasattr(self, "_force_numpy"):
            self._force_numpy = \
                self.__class__.__name__ in args.force_numpy.split(',')
        self._sync = args.sync_run
        self._kernel_ = None
        self._backend_run_ = None
        self.initialize = self._with_backend_init(self.initialize)
        self._numpy_run_jitted_ = False
        if hasattr(self, "numpy_run"):
            # Attribute may be missing if INumpyUnit is not implemented
            self.numpy_run = type(self).numpy_run.__get__(self, type(self))

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value is None:
            raise ValueError("device may not be None")
        if not isinstance(value, Device):
            raise TypeError(
                "device must be of type %s (got %s)" % (Device, type(value)))
        self._device = value

        self.device.assign_backend_methods(self, self.backend_methods)

        if self._sync and self.device.is_async:
            self._original_run_ = self._backend_run_
            self._backend_run_ = self._run_synchronized

    def _run_synchronized(self):
        ret = self._original_run_()
        self.device.sync()
        return ret

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        if not isinstance(value, bool):
            raise TypeError("cache must be boolean (got %s)" % type(value))
        self._cache = value

    @property
    def force_numpy(self):
        return self._force_numpy

    @force_numpy.setter
    def force_numpy(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "force_numpy must be boolean (got %s)" % type(value))
        self._force_numpy = value

    @property
    def sync(self):
        return self._sync

    def initialize(self, device, **kwargs):
        if device is None:
            raise ValueError("device may not be None")
        if not isinstance(device, Device):
            raise TypeError("deviec must be of type %s" % Device)
        if self._force_numpy:
            device = NumpyDevice()

        # Scan class hierarchy
        checked = []  # this is just for exception message
        for cls in type(device).mro():
            if not hasattr(cls, "BACKEND"):
                continue
            checked.append(cls)
            try:
                self.verify_interface(INTERFACE_MAPPING[cls])
                break
            except NotImplementedError:
                pass
        else:
            raise NotImplementedError("%s does not implement any of %s" %
                                      (type(self), checked))

        if not device.is_attached(self.thread_pool):
            device.thread_pool_attach(self.thread_pool)
        try:
            super(AcceleratedUnit, self).initialize(**kwargs)
        except AttributeError:
            pass
        self.device = device
        self.intel_opencl_workaround = (
            self.intel_opencl_workaround and
            isinstance(device, OpenCLDevice) and
            device.device_info.is_cpu)
        if isinstance(self.device, NumpyDevice) and \
                not self._numpy_run_jitted_ and \
                not root.common.engine.disable_numba:
            if jit is None and root.common.warnings.numba:
                self.warning(
                    "Numba (http://numba.pydata.org) was not found, "
                    "numpy_run() is going to be slower. Ignore this warning "
                    "by setting root.common.warnings.numba to False.")
            else:
                self.numpy_run = jit(nopython=True, nogil=True)(self.numpy_run)
                self.debug("Jitted numpy_run() with numba")
                self._numpy_run_jitted_ = True

    def run(self):
        return self._backend_run_()

    def numpy_init(self):
        pass

    def numpy_build_program(self):
        pass

    def numpy_get_kernel(self):
        pass

    def numpy_execute_kernel(self):
        pass

    def _after_backend_init(self):
        pass

    @staticmethod
    def init_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--force-numpy", default="", type=str,
                            help="Force these comma separated accelerated "
                                 "units to run using Numpy (that is, disable "
                                 "OpenCL/CUDA/... for them).")
        parser.add_argument(
            "--sync-run", default=False, action="store_true",
            help="Force accelerated units to run synchronously. This option "
                 "is useful for measuring the actual unit run times.")
        return parser

    def build_program(self, defines=None, cache_file_name=None, dtype=None,
                      **kwargs):
        if cache_file_name is None:
            cache_file_name = self.name
        if not isinstance(cache_file_name, str):
            raise ValueError("cache_file_name must be a string")
        if dtype is None:
            dtype = root.common.engine.precision_type
        elif not isinstance(dtype, str):
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        return self._backend_build_program_(
            defines, cache_file_name, dtype, kwargs)

    def _load_binary(self, defines, cache_file_name, dtype, engine, suffix,
                     cache_is_valid, template_kwargs):
        cache_file_name = "%s.%s.%d" % (cache_file_name, suffix, (2, 3)[PY3])
        if not os.path.isabs(cache_file_name):
            cache_file_name = os.path.join(root.common.dirs.cache,
                                           cache_file_name)
        cache_file_name = "%s.cache" % cache_file_name
        if self.cache and os.path.exists(cache_file_name):
            return self._load_from_cache(
                cache_file_name, defines, dtype, engine, suffix,
                cache_is_valid, template_kwargs)
        else:
            return None, defines

    def ocl_build_program(self, defines, cache_file_name, dtype,
                          template_kwargs):
        """Builds the OpenCL program.

        `program_` will be initialized to the resulting program object.
        """

        def cache_is_valid(cache):
            return (self.device.queue_.device.name ==
                    cache["devices"][0][0] and
                    self.device.queue_.device.platform.name ==
                    cache["devices"][0][1])

        binaries, my_defines = self._load_binary(
            defines, cache_file_name, dtype, "ocl", "cl", cache_is_valid,
            template_kwargs)
        if binaries is not None:
            self.program_ = self.device.queue_.context.create_program(
                binaries, binary=True)
            self._log_about_cache(cache_file_name, "ocl")
            return my_defines
        include_dirs = self._get_include_dirs("ocl")
        source, my_defines = self._generate_source(
            defines, include_dirs, dtype, "cl", template_kwargs)
        show_logs = self.logger.isEnabledFor(logging.DEBUG)
        if show_logs:
            self.debug("%s: source code\n%s\n%s", cache_file_name, "-" * 80,
                       source)
        try:
            self.program_ = self.device.queue_.context.create_program(
                source, include_dirs,
                "-cl-nv-verbose" if show_logs and "cl_nv_compiler_options"
                in self.device.queue_.device.extensions else "")
        except Exception as e:
            with NamedTemporaryFile(mode="w", prefix="ocl_src_", suffix=".cl",
                                    delete=False) as fout:
                fout.write(source)
                self.error("Failed to build OpenCL program. The input file "
                           "source was dumped to %s", fout.name)
            raise from_none(e)
        if show_logs and len(self.program_.build_logs):
            for s in self.program_.build_logs:
                s = s.strip()
                if not s:
                    continue
                self.debug("Non-empty OpenCL build log encountered: %s", s)
        self._save_to_cache(cache_file_name, "cl", self.program_.source,
                            self.program_.binaries,
                            {"devices": [(d.name, d.platform.name)
                                         for d in self.program_.devices]})
        return my_defines

    def cuda_build_program(self, defines, cache_file_name, dtype,
                           template_kwargs):
        """Builds the OpenCL program.

        `program_` will be initialized to the resulting program object.
        """

        def cache_is_valid(cache):
            return self.device.context.device.name == cache["device"]

        binaries, my_defines = self._load_binary(
            defines, cache_file_name, dtype, "cuda", "cu", cache_is_valid,
            template_kwargs)
        if binaries is not None:
            self.program_ = self.device.context.create_module(ptx=binaries)
            self._log_about_cache(cache_file_name, "cuda")
            return my_defines
        include_dirs = self._get_include_dirs("cuda")
        source, my_defines = self._generate_source(
            defines, include_dirs, dtype, "cu", template_kwargs)
        show_logs = self.logger.isEnabledFor(logging.DEBUG)
        if show_logs:
            self.debug("%s: source code\n%s\n%s", cache_file_name, "-" * 80,
                       source)
        try:
            self.program_ = self.device.context.create_module(
                source=source, include_dirs=include_dirs,
                nvcc_path=root.common.engine.cuda.nvcc)
        except Exception as e:
            with NamedTemporaryFile(mode="w", prefix="cuda_src_", suffix=".cu",
                                    delete=False) as fout:
                fout.write(source)
                self.error("Failed to build CUDA program. The input file "
                           "source was dumped to %s", fout.name)
            raise from_none(e)
        if show_logs and len(self.program_.stderr):
            self.debug("Non-empty CUDA build log encountered: %s",
                       self.program_.stderr)
        self._save_to_cache(cache_file_name, "cu", source.encode("utf-8"),
                            self.program_.ptx,
                            {"device": self.device.context.device.name})
        return my_defines

    def _log_about_cache(self, cache_name, engine):
        self.debug('Used the cached %s for engine "%s"', cache_name, engine)

    def ocl_get_kernel(self, name):
        return self.program_.get_kernel(name)

    def cuda_get_kernel(self, name):
        return self.program_.create_function(name)

    def get_kernel(self, name):
        return self._backend_get_kernel_(name)

    def assign_kernel(self, name):
        self._kernel_ = self.get_kernel(name)

    def execute_kernel(self, global_size, local_size, kernel=None,
                       need_event=False):
        try:
            return self._backend_execute_kernel_(
                kernel or self._kernel_, global_size, local_size,
                need_event=need_event)
        except RuntimeError as e:
            self.error("execute_kernel(%s) has failed. global_size = %s, "
                       "local_size = %s", str(kernel or self._kernel_),
                       str(global_size), str(local_size))
            raise from_none(e)

    def ocl_execute_kernel(self, kernel, global_size, local_size, need_event):
        return self.device.queue_.execute_kernel(
            kernel, global_size, local_size, need_event=need_event)

    def cuda_execute_kernel(self, kernel, global_size, local_size,
                            need_event):
        return kernel(global_size,
                      (1, 1, 1) if local_size is None else local_size)

    def set_arg(self, index, arg, kernel=None):
        if kernel is None:
            kernel = self._kernel_
        if isinstance(arg, Array):
            kernel.set_arg(index, arg.devmem)
        else:
            kernel.set_arg(index, arg)

    def set_args(self, *args, **kwargs):
        kernel = kwargs.get("kernel", self._kernel_)
        filtered_args = []
        for arg in args:
            if isinstance(arg, Array):
                filtered_args.append(arg.devmem)
            else:
                filtered_args.append(arg)
        kernel.set_args(*filtered_args)

    def init_vectors(self, *vecs):
        for vec in vecs:
            vec.initialize(self.device)

    def unmap_vectors(self, *vecs):
        for vec in vecs:
            vec.unmap()

    def _get_include_dirs(self, engine):
        return tuple(os.path.join(d, engine)
                     for d in root.common.engine.source_dirs)

    def _adjust_defines(self, my_defines, dtype):
        my_defines.update(opencl_types.cl_defines[dtype])
        if "PRECISION_LEVEL" not in my_defines:
            my_defines["PRECISION_LEVEL"] = root.common.engine.precision_level
        if "GPU_FORCE_64BIT_PTR" not in my_defines:  # for AMD
            my_defines["GPU_FORCE_64BIT_PTR"] = os.getenv(
                "GPU_FORCE_64BIT_PTR", 1)

    def _include_file(self, include_dirs, file, lines):
        for include_dir in include_dirs:
            path = os.path.join(include_dir, file)
            if not os.access(path, os.R_OK):
                continue
            lines.append("\n// #include \"%s\"\n" % file)
            with open(path, "r") as fin:
                lines.extend(fin.readlines())
            lines.append("\n// END OF \"%s\"\n" % file)
            break
        else:
            raise IncludeError("Unable to include \"%s\"" % file)

    def _generate_source(self, defines, include_dirs, dtype, suffix,
                         template_kwargs):
        if defines and not isinstance(defines, dict):
            raise RuntimeError("defines must be a dictionary")
        jsuffix = ".j" + suffix
        suffix = "." + suffix
        lines = []

        def define(cdefs, undef=False):
            for key, value in sorted(cdefs.items()):
                if not undef:
                    lines.append("#define %(key)s %(value)s\n" % locals())
                else:
                    lines.append("#undef %(key)s\n" % locals())

        my_defines = copy(defines) if defines else {}
        self._adjust_defines(my_defines, dtype)
        define(my_defines)
        for name, defs in sorted(self.sources_.items()):
            define(defs)
            if len(template_kwargs) == 0:
                # No templating
                lines.append("#include \"%s%s\"\n" % (name, suffix))
            else:
                try:
                    self._include_file(include_dirs, name + jsuffix, lines)
                except IncludeError:
                    try:
                        self._include_file(include_dirs, name + suffix, lines)
                    except IncludeError:
                        raise from_none(
                            IncludeError("Unable to include \"%s(%s|%s)\"" %
                                         (name, jsuffix, suffix)))
            define(defs, undef=True)
            lines.append("\n")
        source = "".join(lines)
        if len(template_kwargs) == 0:
            return source, my_defines
        include_re = re.compile(
            r'^\s*#\s*include\s*(<(\w+%(sfx)s)>|"(\w+%(sfx)s)")\s*$' %
            {"sfx": "\\" + jsuffix}, flags=re.MULTILINE)
        match = include_re.search(source)
        while match is not None:
            file = match.group(2) or match.group(3)
            lines = []
            self._include_file(include_dirs, file, lines)
            source = include_re.sub("\n" + "".join(lines), source, count=1)
            match = include_re.search(source)
        try:
            source = Template(source).render(**template_kwargs)
        except TemplateError as e:
            self.error(
                "Failed to render the template. Here is the source:\n%s\n",
                "".join("%04d\t%s" % (i + 1, l)
                        for i, l in enumerate(lines)))
            raise from_none(e)
        return source, my_defines

    def _search_include(self, file_name):
        if os.path.exists(file_name):
            return os.path.abspath(file_name)
        for d in root.common.engine.source_dirs:
            full = os.path.join(d + "/" + self.device.backend_name, file_name)
            if os.path.exists(full):
                return os.path.abspath(full)
        return None

    def _search_jinclude(self, file_name, suffix):
        return self._search_include("%s.j%s" % (file_name, suffix)) or \
            self._search_include("%s.%s" % (file_name, suffix))

    def _scan_include_dependencies(self, suffix):
        res = [r for r in (self._search_jinclude(f, suffix)
                           for f in self.sources_.keys())
               if r is not None]
        pending = copy(res)
        include_matcher = re.compile(b'#\s*include\s*\"([\w\.]+)\"')
        while len(pending):
            current = pending.pop(0)
            try:
                with open(current, "rb") as fr:
                    contents = fr.read()
            except:
                self.exception("Failed to read %s", current)
                raise
            for match in include_matcher.finditer(contents):
                header = match.group(1)
                full = self._search_include(header.decode('utf-8'))
                if not full:
                    self.warning("Could not find the header \"%s\" "
                                 "required from %s", header, current)
                    continue
                pending.append(full)
                res.append(full)
        return res

    def _load_from_cache(self, cache_file_name, defines, dtype, engine, suffix,
                         cache_is_valid, template_kwargs):
        include_dirs = self._get_include_dirs(engine)
        try:
            with tarfile.open(cache_file_name, "r:gz") as tar:
                cached_source = tar.extractfile("source." + suffix).read()
                src, my_defines = self._generate_source(
                    defines, include_dirs, dtype, suffix, template_kwargs)
                real_source = src.encode("utf-8")
                if cached_source != real_source:
                    return None, None
                for dep in set(self._scan_include_dependencies(suffix)):
                    cached_source = tar.extractfile(
                        os.path.basename(dep)).read()
                    with open(dep, "rb") as fr:
                        real_source = fr.read()
                    if cached_source != real_source:
                        return None, None
                cache = pickle.loads(tar.extractfile("binaries.pickle").read())
                if not cache_is_valid(cache):
                    return None, None
                bins = cache["binaries"]
                if not isinstance(bins, bytes) and (
                        not isinstance(bins, list) or len(bins) == 0 or
                        not isinstance(bins[0], bytes)):
                    self.warning("Cached binaries have an invalid format")
                    return None, None
                return cache["binaries"], my_defines
        except Exception as e:
            self.debug("Failed to load %s: %s", cache_file_name, e)
            return None, None

    def _save_to_cache(self, cache_file_name, suffix, program_source,
                       program_binaries, device_id_dict):
        naked_suffix = suffix
        suffix = "." + suffix
        cache_file_name = cache_file_name + suffix + (".3" if PY3 else ".2")
        if not os.path.isabs(cache_file_name):
            cache_file_name = os.path.join(root.common.dirs.cache,
                                           cache_file_name)
        try:
            with tarfile.open("%s.cache" % cache_file_name, "w:gz") as tar:
                source_io = BytesIO()
                source_io.write(program_source)
                ti = tarfile.TarInfo("source" + suffix)
                ti.size = source_io.tell()
                ti.mode = int("666", 8)
                source_io.seek(0)
                tar.addfile(ti, fileobj=source_io)
                for dep in set(self._scan_include_dependencies(naked_suffix)):
                    ti = tarfile.TarInfo(os.path.basename(dep))
                    ti.size = os.path.getsize(dep)
                    ti.mode = int("666", 8)
                    with open(dep, "rb") as fr:
                        tar.addfile(ti, fileobj=fr)
                binaries_io = BytesIO()
                pickler = pickle.Pickler(binaries_io, protocol=best_protocol)
                binaries = {"binaries": program_binaries}
                binaries.update(device_id_dict)
                pickler.dump(binaries)
                ti = tarfile.TarInfo("binaries.pickle")
                ti.size = binaries_io.tell()
                ti.mode = int("666", 8)
                binaries_io.seek(0)
                tar.addfile(ti, fileobj=binaries_io)
        except:
            self.exception("Failed to save the cache file %s:",
                           cache_file_name)

    def _with_backend_init(self, fn):
        def wrapped_backend_init(device, **kwargs):
            result = fn(device, **kwargs)
            if not result:
                self._backend_init_()
                self._after_backend_init()
            return result

        wrapped_backend_init.__name__ = fn.__name__ + "_backend_init"
        return wrapped_backend_init


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class TrivialAcceleratedUnit(AcceleratedUnit):
    def numpy_run(self):
        pass

    def ocl_run(self):
        pass

    def cuda_run(self):
        pass

    def ocl_init(self):
        pass

    def cuda_init(self):
        pass


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class DeviceBenchmark(AcceleratedUnit):
    """
    Executes an OpenCL benchmark to estimate the computing power of the device.
    """

    def __init__(self, workflow, **kwargs):
        super(DeviceBenchmark, self).__init__(workflow, **kwargs)
        self.precision = kwargs.get("dtype", root.common.engine.precision_type)
        self.dtype = opencl_types.dtypes[self.precision]
        self.size = kwargs.get("size", 1500)
        self.repeats = kwargs.get("repeats", 10)
        self._input_A_ = Array()
        self._input_B_ = Array()
        msize = self.size * self.size
        from veles.prng.random_generator import RandomGenerator
        rnd = RandomGenerator(None)
        genmem = lambda: rnd.rand(msize).astype(self.dtype) - 0.5
        self._input_A_.mem = genmem()
        self._input_B_.mem = genmem()
        self.block_size = kwargs.get("block_size")
        self.vector_opt = kwargs.get("vector_opt")
        self.precision_level = kwargs.get("precision_level",
                                          root.common.engine.precision_level)
        self.return_time = kwargs.get("return_time", False)
        self.dry_run_first = kwargs.get("dry_run_first", False)

    def initialize(self, device, **kwargs):
        """Compiles the benchmarking kernel.
        """
        super(DeviceBenchmark, self).initialize(device=device, **kwargs)
        if isinstance(device, NumpyDevice):
            self._input_A_.mem = self._input_A_.mem.reshape(
                self.size, self.size)
            self._input_B_.mem = self._input_B_.mem.reshape(
                self.size, self.size)

    def ocl_init(self):
        if self.block_size is None or self.vector_opt is None:
            bs_vo = self.device.device_info.get_kernel_bs_vo(
                kernel="matrix_multiplication", dtype=self.precision)
        if self.block_size is None:
            self.block_size = bs_vo[0]
        if self.vector_opt is None:
            self.vector_opt = bs_vo[1]
        self.sources_["benchmark"] = {}
        defines = {
            "BLOCK_SIZE": self.block_size,
            "VECTOR_OPT": int(bool(self.vector_opt)),
            "SIZE": self.size,
            "PRECISION_LEVEL": self.precision_level}
        self.build_program(defines, dtype=self.precision)
        self.assign_kernel("benchmark")
        self.init_vectors(self._input_A_, self._input_B_)
        self.set_args(self._input_A_, self._input_A_, self._input_B_)

    def cuda_init(self):
        import cuda4py.blas as cublas
        self.gemm_ = cublas.CUBLAS.gemm(self.dtype)
        self.np_one = numpy.ones(1, self.dtype)
        self.np_zero = numpy.zeros(1, self.dtype)
        self.init_vectors(self._input_A_, self._input_B_)

    def run(self):
        self.debug("Running %d repetitions of size %d on %s...",
                   self.repeats, self.size, self.precision)
        dt = super(DeviceBenchmark, self).run()
        if self.return_time:
            res = dt / self.repeats
            self.debug("Avg time is %.6f", res)
        else:
            res = 1000 / dt
            self.debug("Result is %.2f", res)
        return res

    def numpy_run(self):
        def execute(repeats):
            for _ in range(repeats):
                numpy.dot(self._input_A_.mem, self._input_A_.mem,
                          self._input_B_.mem)

        if self.dry_run_first:
            execute(1)

        return timeit(execute, self.repeats)[1]

    def ocl_run(self):
        global_size = (roundup(self.size, self.block_size),) * 2
        local_size = (self.block_size,) * 2
        self.device.queue_.flush()
        self.device.queue_.finish()

        def execute(repeats):
            for _ in range(repeats):
                self.execute_kernel(global_size, local_size)
            self.device.queue_.flush()
            self.device.queue_.finish()

        if self.dry_run_first:
            execute(1)

        return timeit(execute, self.repeats)[1]

    def cuda_run(self):
        import cuda4py.blas as cublas
        self.device.sync()

        def execute(repeats):
            for _ in range(repeats):
                self.gemm_(
                    self.device.blas, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                    self.size, self.size, self.size,
                    self.np_one, self._input_A_.devmem, self._input_A_.devmem,
                    self.np_zero, self._input_B_.devmem)
            self.device.sync()

        if self.dry_run_first:
            execute(1)

        return timeit(execute, self.repeats)[1]


class AcceleratedWorkflow(Workflow):
    """Base class for OpenCL workflows.
    """

    def __init__(self, workflow, **kwargs):
        super(AcceleratedWorkflow, self).__init__(workflow, **kwargs)
        self._power_measure_time_interval = kwargs.get(
            'power_measure_time_interval', 120)

    def init_unpickled(self):
        super(AcceleratedWorkflow, self).init_unpickled()
        self._power_ = 0
        self._last_power_measurement_time = 0
        self.device = NumpyDevice()
        self._power_measure_time_interval = 120

    @property
    def computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        now = time.time()
        if (now - self._last_power_measurement_time >
                self._power_measure_time_interval):
            self._last_power_measurement_time = now
            with self:
                bench = DeviceBenchmark(self)
                bench.initialize(self.device)
                self._power_ = bench.run()
            self.info("Computing power is %.2f", self._power_)
        return self._power_

    def initialize(self, device, **kwargs):
        super(AcceleratedWorkflow, self).initialize(device=device, **kwargs)
        self.device = device

    def filter_unit_graph_attrs(self, val):
        return (not isinstance(val, Device) and
                super(AcceleratedWorkflow, self).filter_unit_graph_attrs(val))
