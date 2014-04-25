'''
Created on Apr 25, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
'''


from copy import copy
import numpy
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
        cl_sources: OpenCL source files: file => defines.
        prg_src: last built OpenCL program source code text.
    """
    def __init__(self, workflow, **kwargs):
        super(OpenCLUnit, self).__init__(workflow, **kwargs)
        self.device = None

    def init_unpickled(self):
        super(OpenCLUnit, self).init_unpickled()
        self.program_ = None
        self.cl_sources_ = {}

    def initialize(self, device, **kwargs):
        super(OpenCLUnit, self).initialize(device=device, **kwargs)
        self.device = device or self.workflow.device

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

    def build_program(self, defines=None, dump_filename=None, dtype=None):
        """Builds the OpenCL program.

        program_ will be initialized to the resulting program object.
        """
        if defines and not isinstance(defines, dict):
            raise RuntimeError("defines must be a dictionary")
        lines = []
        my_defines = copy(defines) if defines else {}
        for fnme, defs in self.cl_sources_.items():
            lines.append("#include \"%s\"" % (fnme))
            my_defines.update(defs)
        if dtype is None:
            dtype = root.common.precision_type
        elif type(dtype) != str:
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        my_defines.update(opencl_types.cl_defines[dtype])

        for k, v in my_defines.items():
            lines.insert(0, "#define %s %s" % (k, v))

        source = "\n".join(lines)

        try:
            self.program_ = self.device.queue_.context.create_program(
                source, root.common.ocl_dirs)
            if len(self.program_.build_logs):
                for s in self.program_.build_logs:
                    s = s.strip()
                    if not len(s):
                        continue
                    self.info("Non-empty OpenCL build log encountered: %s", s)
        finally:
            if dump_filename is not None:
                flog = open(dump_filename, "w")
                flog.write(source)
                flog.close()

    def get_kernel(self, name):
        return self.program_.get_kernel(name)

    def execute_kernel(self, krn, global_size, local_size):
        return self.device.queue_.execute_kernel(krn, global_size, local_size)


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
        self._power = None

    @property
    def computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        if not self._power:
            bench = OpenCLBenchmark(self, device=self.device)
            self._power = bench.estimate()
            self.del_ref(bench)
            self.info("Computing power is %.6f", self._power)
        return self._power
