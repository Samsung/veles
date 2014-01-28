"""
Created on Jan 28, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import numpy
import pyopencl
import time

import formats
import opencl
import units


class OpenCLBenchmark(units.OpenCLUnit):
    """
    Executes an OpenCL benchmark to estimate the computing power.
    """

    def __init__(self, device=None):
        if device == None:
            device = opencl.Device()
        super(OpenCLBenchmark, self).__init__(device=device)
        self.block_size = 30
        self.size = 3000
        self.cl_sources_ = {"benchmark.cl": {
            'BLOCK_SIZE': self.block_size,
            'SIZE': self.size
        }}
        self.build_program()
        self.kernel_ = self.get_kernel("benchmark")
        msize = [self.size, self.size]
        self.input_A_ = formats.Vector()
        self.input_B_ = formats.Vector()
        self.output_C_ = formats.Vector()
        self.input_A_.v = numpy.zeros(msize, dtype=numpy.double)
        self.input_B_.v = numpy.zeros(msize, dtype=numpy.double)
        self.output_C_.v = numpy.zeros(msize, dtype=numpy.double)
        self.input_A_.initialize(device)
        self.input_B_.initialize(device)
        self.output_C_.initialize(device)
        self.kernel_.set_arg(0, self.input_A_.v_)
        self.kernel_.set_arg(1, self.input_B_.v_)
        self.kernel_.set_arg(2, self.output_C_.v_)

    def estimate(self):
        """
        Launches and waits for the benchmark to finish being executed.
        """
        global_size = [formats.roundup(self.size, self.block_size),
                       formats.roundup(self.size, self.block_size)]
        local_size = [self.block_size, self.block_size]
        tstart = time.time()
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.kernel_,
                                                 global_size, local_size)
        event.wait()
        self.output_C_.map_read()
        tfinish = time.time()
        delta = tfinish - tstart
        return 1000 / delta
