'''
Created on Apr 25, 2014

Units that use hardware acceleration, either OpenCL, CUDA or other supported
backend.

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
'''


import argparse
from copy import copy
from jinja2 import Template, TemplateError
import logging
import numpy
import os
import re
from six import BytesIO, add_metaclass, PY3
import tarfile
import time
from zope.interface import implementer, Interface
from veles.compat import from_none

from veles.config import root
from veles.memory import Vector, roundup
import veles.opencl_types as opencl_types
from veles.backends import Device, OpenCLDevice
from veles.pickle2 import pickle, best_protocol
from veles.timeit import timeit
from veles.units import Unit, IUnit, UnitCommandLineArgumentsRegistry
from veles.workflow import Workflow


class IncludeError(Exception):
    pass


class IOpenCLUnit(Interface):
    """Requires cpu and ocl run() methods for OpenCLUnit.
    """

    def cpu_run():
        """Run on CPU.
        """

    def ocl_run():
        """Run on GPU/any OpenCL capable device.
        """

    def initialize(device, **kwargs):
        """initialize() with "device" obligatory argument.
        """


@implementer(IUnit)
@add_metaclass(UnitCommandLineArgumentsRegistry)
class AcceleratedUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        program_: OpenCL program.
        sources_: OpenCL source files: file => defines.
        cache: whether to cache the compiled OpenCL programs.
    """
    backend_methods = ("run", "init", "build_program", "get_kernel",
                       "execute_kernel")
    hide = True

    def __init__(self, workflow, **kwargs):
        super(AcceleratedUnit, self).__init__(workflow, **kwargs)
        self.verify_interface(IOpenCLUnit)
        self._device = None
        self._cache = kwargs.get("cache", True)
        # Yup, this is right - self._force_cpu is initialized in init_unpickled
        self._force_cpu = kwargs.get("force_cpu", self._force_cpu)
        self.prefer_numpy = root.common.prefer_numpy_on_cpu

    def init_unpickled(self):
        super(AcceleratedUnit, self).init_unpickled()
        self.program_ = None
        self.sources_ = {}
        parser = AcceleratedUnit.init_parser()
        args, _ = parser.parse_known_args()
        self._force_cpu = self.__class__.__name__ in args.force_cpu.split(',')
        self._sync = args.sync_ocl
        self._kernel_ = None
        self._backend_run_ = None
        self.initialize = self.with_backend_init(self.initialize)

    def with_backend_init(self, fn):
        def wrapped(device, **kwargs):
            result = fn(device, **kwargs)
            if not result:
                self._backend_init_()
            return result

        wrapped.__name__ = fn.__name__ + "_backend_init"
        return wrapped

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if not isinstance(value, Device) and value is not None:
            raise TypeError("device must be of type veles.opencl.Device (%s "
                            "was specified)" % value.__class__)
        self._device = value
        if value is None or self.force_cpu:
            backend = "cpu"
        else:
            backend = value.backend_name
        for suffix in self.backend_methods:
            setattr(self, "_backend_" + suffix + "_",
                    getattr(self, backend + "_" + suffix))
        if self._sync and value is not None:
            self._original_run_ = self._backend_run_
            self._backend_run_ = self._run_synchronized

    def _run_synchronized(self):
        self._original_run_()
        self.device.sync()

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def force_cpu(self):
        return self._force_cpu

    @force_cpu.setter
    def force_cpu(self, value):
        self._force_cpu = value

    def initialize(self, device, **kwargs):
        try:
            super(AcceleratedUnit, self).initialize(**kwargs)
        except AttributeError:
            pass
        self.device = device
        if not (self.device is None or self.device.exists or self._force_cpu):
            self.critical(
                "No device exists and --disable-acceleration option was not "
                "specified")
            raise ValueError("No device was found")
        if self._force_cpu:
            self.device = None
        # TODO(a.kazantsev): remove prefer_numpy.
        self.prefer_numpy = (self.prefer_numpy and self.device is not None and
                             (isinstance(device, OpenCLDevice) and
                              device.device_info.is_cpu))

    def cpu_init(self):
        pass

    def run(self):
        return self._backend_run_()

    @property
    def sync(self):
        return self._sync

    @staticmethod
    def init_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--force-cpu", default="", type=str,
                            help="Force these comma separated accelerated "
                                 "units to run on CPU (that is, disable "
                                 "OpenCL/CUDA/... for them).")
        parser.add_argument("--sync-ocl", default=False, action="store_true",
                            help="Force OpenCL units to run synchronously. "
                            "This option is useful for measuring the actual "
                            "unit run times.")
        return parser

    def build_program(self, defines=None, cache_file_name=None, dtype=None,
                      **kwargs):
        if cache_file_name is None:
            cache_file_name = self.name
        if not isinstance(cache_file_name, str):
            raise ValueError("cache_file_name must be a string")
        if dtype is None:
            dtype = root.common.precision_type
        elif type(dtype) != str:
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        return self._backend_build_program_(
            defines, cache_file_name, dtype, kwargs)

    def cpu_build_program(self, defines, cache_file_name, dtype):
        pass

    def _load_binary(self, defines, cache_file_name, dtype, engine, suffix,
                     cache_is_valid, template_kwargs):
        cache_file_name = cache_file_name + suffix + (".3" if PY3 else ".2")
        if not os.path.isabs(cache_file_name):
            cache_file_name = os.path.join(root.common.cache_dir,
                                           cache_file_name)
        if self.cache and os.path.exists("%s.cache" % cache_file_name):
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
            self.debug("Used %s.cache", cache_file_name)
            return my_defines
        include_dirs = [os.path.join(d, "ocl")
                        for d in root.common.engine.dirs]
        source, my_defines = self._generate_source(
            defines, include_dirs, dtype, "cl", template_kwargs)
        show_ocl_logs = self.logger.isEnabledFor(logging.DEBUG)
        self.program_ = self.device.queue_.context.create_program(
            source, include_dirs,
            "-cl-nv-verbose" if show_ocl_logs and "cl_nv_compiler_options" in
            self.device.queue_.device.extensions else "")
        if show_ocl_logs and len(self.program_.build_logs):
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
            self.debug("Used %s.cache", cache_file_name)
            return my_defines
        include_dirs = [os.path.join(d, "cuda")
                        for d in root.common.engine.dirs]
        source, my_defines = self._generate_source(
            defines, include_dirs, dtype, "cu", template_kwargs)
        show_logs = self.logger.isEnabledFor(logging.DEBUG)
        self.program_ = self.device.context.create_module(
            source=source, include_dirs=include_dirs)
        if show_logs and len(self.program_.stderr):
            self.debug("Non-empty CUDA build log encountered: %s",
                       self.program_.stderr)
        self._save_to_cache(cache_file_name, "cu", source.encode("utf-8"),
                            self.program_.ptx,
                            {"device": self.device.context.device.name})
        return my_defines

    def ocl_get_kernel(self, name):
        return self.program_.get_kernel(name)

    def cuda_get_kernel(self, name):
        return self.program_.create_function(name)

    def cpu_get_kernel(self, name):
        return None

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
        except RuntimeError:
            self.error("execute_kernel(%s) has failed. global_size = %s, "
                       "local_size = %s", str(kernel or self._kernel_),
                       str(global_size), str(local_size))
            raise

    def cpu_execute_kernel(self, kernel, global_size, local_size, need_event):
        return None

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
        if isinstance(arg, Vector):
            kernel.set_arg(index, arg.devmem)
        else:
            kernel.set_arg(index, arg)

    def set_args(self, *args, **kwargs):
        kernel = kwargs.get("kernel", self._kernel_)
        filtered_args = []
        for arg in args:
            if isinstance(arg, Vector):
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

    def _adjust_defines(self, my_defines, dtype):
        my_defines.update(opencl_types.cl_defines[dtype])
        if "PRECISION_LEVEL" not in my_defines:
            my_defines["PRECISION_LEVEL"] = root.common.precision_level
        if ("VECTOR_OPT" not in my_defines and
                hasattr(self.device, "device_info") and
                hasattr(self.device.device_info, "vector_opt")):
            my_defines["VECTOR_OPT"] = self.device.device_info.vector_opt
        if "GPU_FORCE_64BIT_PTR" not in my_defines:  # for AMD
            my_defines["GPU_FORCE_64BIT_PTR"] = os.getenv(
                "GPU_FORCE_64BIT_PTR", 0)

    def _include_file(self, include_dirs, file, lines):
        for include_dir in include_dirs:
            path = os.path.join(include_dir, file)
            if not os.access(path, os.R_OK):
                continue
            lines.append("\n// #include \"%s\"\n" % file)
            with open(path, "r") as fin:
                lines.extend(fin.readlines())
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
                    lines.insert(0, "#define %(key)s %(value)s\n" % locals())
                else:
                    lines.append("#undef %(key)s\n" % locals())

        my_defines = copy(defines) if defines else {}
        for name, defs in self.sources_.items():
            define(defs)
            if len(template_kwargs) == 0:
                # No templating
                lines.append("#include \"%s%s\"\n" % (name, suffix))
                continue
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
        self._adjust_defines(my_defines, dtype)
        define(my_defines)
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
        for d in root.common.engine.dirs:
            full = os.path.join(d + "/" + self.device.backend_name, file_name)
            if os.path.exists(full):
                return os.path.abspath(full)
        return ""

    def _scan_include_dependencies(self, suffix):
        res = [self._search_include(f + suffix)
               for f in self.sources_.keys()]
        pending = copy(res)
        include_matcher = re.compile(b'#\s*include\s*\"([\w\.]+)\"')
        while len(pending):
            try:
                with open(pending[0], "rb") as fr:
                    contents = fr.read()
            except:
                self.exception("Failed to read %s", pending[0])
                raise
            for match in include_matcher.finditer(contents):
                header = match.group(1)
                full = self._search_include(header.decode('utf-8'))
                if not full:
                    self.warning("Could not find the header \"%s\" "
                                 "required from %s", header, pending[0])
                    continue
                pending.append(full)
                res.append(full)
            pending = pending[1:]
        return res

    def _load_from_cache(self, cache_file_name, defines, dtype, engine, suffix,
                         cache_is_valid, template_kwargs):
        include_dirs = tuple(os.path.join(d, engine)
                             for d in root.common.engine.dirs)
        try:
            with tarfile.open("%s.cache" % cache_file_name, "r:gz") as tar:
                cached_source = tar.extractfile("source" + suffix).read()
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
        suffix = "." + suffix
        cache_file_name = cache_file_name + suffix + (".3" if PY3 else ".2")
        if not os.path.isabs(cache_file_name):
            cache_file_name = os.path.join(root.common.cache_dir,
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
                for dep in set(self._scan_include_dependencies(suffix)):
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


@implementer(IOpenCLUnit)
class TrivialOpenCLUnit(AcceleratedUnit):
    def cpu_run(self):
        pass

    def ocl_run(self):
        pass

    def cuda_run(self):
        pass

    def ocl_init(self):
        pass

    def cuda_init(self):
        pass


@implementer(IOpenCLUnit)
class DeviceBenchmark(AcceleratedUnit):
    """
    Executes an OpenCL benchmark to estimate the computing power of the device.
    """

    def __init__(self, workflow, **kwargs):
        super(DeviceBenchmark, self).__init__(workflow, **kwargs)
        self.dtype = kwargs.get("dtype", root.common.precision_type)
        dtype = opencl_types.dtypes[self.dtype]
        self.size = kwargs.get("size", 1500)
        self.repeats = kwargs.get("repeats", 10)
        self.input_A_ = Vector()
        self.input_B_ = Vector()
        msize = self.size * self.size
        genmem = lambda: numpy.random.rand(msize).astype(dtype) - 0.5
        self.input_A_.mem = genmem()
        self.input_B_.mem = genmem()
        self.block_size = kwargs.get("block_size")
        self.precision_level = kwargs.get("precision_level",
                                          root.common.precision_level)

    def initialize(self, device, **kwargs):
        """Compiles the benchmarking kernel.
        """
        super(DeviceBenchmark, self).initialize(device=device, **kwargs)
        if device is None:
            self.input_A_.mem = self.input_A_.mem.reshape(self.size, self.size)
            self.input_B_.mem = self.input_B_.mem.reshape(self.size, self.size)
            return

    def ocl_init(self):
        if self.block_size is None and self.device is not None:
            self.block_size = self.device.device_info.get_block_size(
                kernel="matrix_multiplication", dtype=self.dtype)
        self.sources_["benchmark"] = {}
        defines = {
            "BLOCK_SIZE": self.block_size,
            "SIZE": self.size,
            "PRECISION_LEVEL": self.precision_level
        }
        self.build_program(defines, dtype=self.dtype)
        self.assign_kernel("benchmark")
        self.input_A_.initialize(self.device)
        self.input_B_.initialize(self.device)
        self.set_args(self.input_A_, self.input_A_, self.input_B_)

    def estimate(self, return_time=False, dry_run_first=False):
        """
        Launches and waits for the benchmark to finish.
        """
        if self.device is not None:
            dt = self._estimate_ocl(dry_run_first)
        else:
            dt = self._estimate_cpu(dry_run_first)
        if return_time:
            res = dt / self.repeats
            self.debug("Avg time is %.6f", res)
        else:
            res = 1000 / dt
            self.debug("Result is %.2f", res)
        return res

    def _estimate_cpu(self, dry_run_first):
        def execute(repeats):
            for _ in range(repeats):
                numpy.dot(self.input_A_.mem, self.input_A_.mem,
                          self.input_B_.mem)

        if dry_run_first:
            execute(1)

        return timeit(execute, self.repeats)[1]

    def _estimate_ocl(self, dry_run_first):
        self.debug("Running %d repetitions of size %d on %s...",
                   self.repeats, self.size, self.dtype)
        global_size = [roundup(self.size, self.block_size),
                       roundup(self.size, self.block_size)]
        local_size = [self.block_size, self.block_size]
        self.device.queue_.flush()
        self.device.queue_.finish()

        def execute(repeats):
            for _ in range(repeats):
                self.execute_kernel(global_size, local_size)
            self.device.queue_.flush()
            self.device.queue_.finish()

        if dry_run_first:
            execute(1)

        return timeit(execute, self.repeats)[1]

    def cpu_run(self):
        self.estimate()

    def ocl_run(self):
        self.estimate()


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
        self.device = None
        # FIXME(v.markovtsev): remove the line below when Lubov's code is sync
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
                self._power_ = bench.estimate()
            self.info("Computing power is %.2f", self._power_)
        return self._power_

    def initialize(self, device, **kwargs):
        super(AcceleratedWorkflow, self).initialize(device=device, **kwargs)
        self.device = device

    def filter_unit_graph_attrs(self, val):
        return (not isinstance(val, Device) and
                super(AcceleratedWorkflow, self).filter_unit_graph_attrs(val))
