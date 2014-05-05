'''
Created on Apr 25, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
'''


from copy import copy
import numpy
import os
import re
from six.moves import cPickle as pickle
from six import BytesIO
from six import PY3
import tarfile
import time

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.units as units
import veles.workflow as workflow


class OpenCLUnit(units.Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        program_: OpenCL program.
        cl_sources_: OpenCL source files: file => defines.
        cache: whether to cache the compiled OpenCL programs.
    """
    def __init__(self, workflow, **kwargs):
        super(OpenCLUnit, self).__init__(workflow, **kwargs)
        self.device = None
        self._cache = kwargs.get("cache", True)

    def init_unpickled(self):
        super(OpenCLUnit, self).init_unpickled()
        self.program_ = None
        self.cl_sources_ = {}

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    def initialize(self, device, **kwargs):
        super(OpenCLUnit, self).initialize(device=device, **kwargs)
        self.device = device

    def run(self):
        if self.device:
            self.ocl_run()
        else:
            self.cpu_run()

    def cpu_run(self):
        """Run on CPU only.
        """
        return super(OpenCLUnit, self).run()

    def ocl_run(self):
        """Run on GPU/any OpenCL capable device.
        """
        return self.cpu_run()

    def build_program(self, defines=None, cache_file_name=None, dtype=None):
        """Builds the OpenCL program.

        program_ will be initialized to the resulting program object.
        """
        if type(cache_file_name) == str:
            cache_file_name = cache_file_name + (".3" if PY3 else ".2")
        if self.cache and os.path.exists("%s.cache" % cache_file_name):
            binaries = self._load_from_cache(cache_file_name, defines, dtype)
            if binaries is not None:
                self.program_ = self.device.queue_.context.create_program(
                    binaries, binary=True)
                self.debug("Used %s.cache", cache_file_name)
                return
        source = self._generate_source(defines, dtype)
        self.program_ = self.device.queue_.context.create_program(
            source, root.common.ocl_dirs)
        if len(self.program_.build_logs):
            for s in self.program_.build_logs:
                s = s.strip()
                if not s:
                    continue
                self.info("Non-empty OpenCL build log encountered: %s", s)
        self._save_to_cache(cache_file_name)

    def get_kernel(self, name):
        return self.program_.get_kernel(name)

    def execute_kernel(self, krn, global_size, local_size):
        return self.device.queue_.execute_kernel(krn, global_size, local_size)

    def _generate_source(self, defines, dtype=None):
        if defines and not isinstance(defines, dict):
            raise RuntimeError("defines must be a dictionary")
        lines = []
        my_defines = copy(defines) if defines else {}
        for fnme, defs in self.cl_sources_.items():
            lines.append("#include \"%s\"" % fnme)
            my_defines.update(defs)
        if dtype is None:
            dtype = root.common.precision_type
        elif type(dtype) != str:
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        my_defines.update(opencl_types.cl_defines[dtype])

        for k, v in sorted(my_defines.items()):
            lines.insert(0, "#define %s %s" % (k, v))

        source = "\n".join(lines)
        return source

    def _search_include(self, file_name):
        if os.path.exists(file_name):
            return os.path.abspath(file_name)
        for d in root.common.ocl_dirs:
            full = os.path.join(d, file_name)
            if os.path.exists(full):
                return os.path.abspath(full)
        return ""

    def _scan_include_dependencies(self):
        res = [self._search_include(f) for f in self.cl_sources_.keys()]
        pending = copy(res)
        include_matcher = re.compile('#\s*include\s*((")?|(<)?)([\w\.]+)'
                                     '(?(2)"|>)')
        while len(pending):
            with open(pending[0], "r") as fr:
                contents = fr.read()
            for match in include_matcher.finditer(contents):
                header = match.group(4)
                full = self._search_include(header)
                if not full:
                    self.warning("Could not find the header \"%s\" "
                                 "required from %s", header, pending[0])
                    continue
                pending.append(full)
                res.append(full)
            pending = pending[1:]
        return res

    def _load_from_cache(self, cache_file_name, defines, dtype):
        try:
            with tarfile.open("%s.cache" % cache_file_name, "r") as tar:
                cached_source = tar.extractfile("source.cl").read()
                real_source = self._generate_source(defines, dtype).encode()
                if cached_source != real_source:
                    return None
                for dep in self._scan_include_dependencies():
                    cached_source = tar.extractfile(
                        os.path.basename(dep)).read()
                    with open(dep, "rb") as fr:
                        real_source = fr.read()
                    if cached_source != real_source:
                        return None
                cache = pickle.loads(tar.extractfile("binaries.pickle").read())
                if (self.device.queue_.device.name != cache["devices"][0][0] or
                    self.device.queue_.device.platform.name !=
                        cache["devices"][0][1]):
                    return None
                return cache["binaries"]
        except:
            return None

    def _save_to_cache(self, cache_file_name):
        with tarfile.open("%s.cache" % cache_file_name, "w") as tar:
            source_io = BytesIO()
            source_io.write(self.program_.source)
            ti = tarfile.TarInfo("source.cl")
            ti.size = source_io.tell()
            ti.mode = int("666", 8)
            source_io.seek(0)
            tar.addfile(ti, fileobj=source_io)
            for dep in self._scan_include_dependencies():
                ti = tarfile.TarInfo(os.path.basename(dep))
                ti.size = os.path.getsize(dep)
                ti.mode = int("666", 8)
                with open(dep, "rb") as fr:
                    tar.addfile(ti, fileobj=fr)
            binaries_io = BytesIO()
            pickler = pickle.Pickler(binaries_io)
            binaries = {"binaries": self.program_.binaries,
                        "devices": [(d.name, d.platform.name)
                                    for d in self.program_.devices]}
            pickler.dump(binaries)
            ti = tarfile.TarInfo("binaries.pickle")
            ti.size = binaries_io.tell()
            ti.mode = int("666", 8)
            binaries_io.seek(0)
            tar.addfile(ti, fileobj=binaries_io)


class OpenCLBenchmark(OpenCLUnit):
    """
    Executes an OpenCL benchmark to estimate the computing power of the device.
    """

    def __init__(self, workflow, **kwargs):
        super(OpenCLBenchmark, self).__init__(workflow, **kwargs)
        self.block_size = 30
        self.size = 3000
        self.cl_sources_ = {"benchmark.cl": {
            'BLOCK_SIZE': self.block_size,
            'SIZE': self.size
        }}
        self.input_A_ = formats.Vector()
        self.input_B_ = formats.Vector()
        self.output_C_ = formats.Vector()
        msize = [self.size, self.size]
        self.input_A_.v = numpy.zeros(msize, dtype=numpy.double)
        self.input_B_.v = numpy.zeros(msize, dtype=numpy.double)
        self.output_C_.v = numpy.zeros(msize, dtype=numpy.double)

    def initialize(self, device, **kwargs):
        """Compiles the benchmarking kernel.
        """
        super(OpenCLBenchmark, self).initialize(device=device, **kwargs)
        self.build_program()
        self.kernel_ = self.get_kernel("benchmark")
        self.input_A_.initialize(self.device)
        self.input_B_.initialize(self.device)
        self.output_C_.initialize(self.device)
        self.kernel_.set_arg(0, self.input_A_.v_)
        self.kernel_.set_arg(1, self.input_B_.v_)
        self.kernel_.set_arg(2, self.output_C_.v_)

    def estimate(self):
        """
        Launches and waits for the benchmark to finish.
        """
        global_size = [formats.roundup(self.size, self.block_size),
                       formats.roundup(self.size, self.block_size)]
        local_size = [self.block_size, self.block_size]
        tstart = time.time()
        event = self.device.queue_.execute_kernel(self.kernel_, global_size,
                                                  local_size)
        event.wait()
        self.output_C_.map_read()
        tfinish = time.time()
        delta = tfinish - tstart
        return 1000 / delta


class OpenCLWorkflow(OpenCLUnit, workflow.Workflow):
    """Base class for OpenCL workflows.
    """

    def __init__(self, workflow, **kwargs):
        super(OpenCLWorkflow, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(OpenCLWorkflow, self).init_unpickled()
        self._power_ = None

    @property
    def computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        if not self._power_:
            bench = OpenCLBenchmark(self, device=self.device)
            self._power_ = bench.estimate()
            self.del_ref(bench)
            self.info("Computing power is %.6f", self._power)
        return self._power_
