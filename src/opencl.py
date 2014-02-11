"""
Created on Mar 21, 2013

OpenCL helper classes.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import os
import pickle
import pyopencl
import sys
import time
import traceback

import config
import error
import formats
import opencl_types
import rnd
import units


class DeviceInfo(object):
    """Info about device.

    Attributes:
        guid: "GUID" of the device.
        memsize: "available" size of the memory on the device.
        memalign: best alignment for device buffers.
        version: OpenCL version.
        rating: in [0, 1] interval (1 - fastest, 0.5 - 50% slower than fastest,
                0 - unrated).
        dt: time of rating test pass.
        min_dt: minimum time of rating test pass of all tests.
        BLOCK_SIZE: best block size for matrix multiplication for the device.
    """
    def __init__(self, guid, memsize, memalign, version):
        self.guid = guid
        self.memsize = memsize
        self.memalign = memalign
        self.version = version
        self.rating = {}
        for dtype in opencl_types.dtypes.keys():
            self.rating[dtype] = 0.0
        self.dt = {}
        for dtype in opencl_types.dtypes.keys():
            self.dt[dtype] = 86400
        self.min_dt = {}
        for dtype in opencl_types.dtypes.keys():
            self.min_dt[dtype] = 86400
        self.BLOCK_SIZE = {}
        for dtype in opencl_types.dtypes.keys():
            self.BLOCK_SIZE[dtype] = 16


class Device(units.Pickleable):
    """OpenCL device helper class.

    Attributes:
        info: DeviceInfo object.
        context_: OpenCL context handle.
        queue_: OpenCL device queue.
        pid_: process id.
    """
    def __init__(self):
        super(Device, self).__init__()
        if not config.plotters_disabled:
            import plotters
            plotters.Plotter(None)  # will register on shutdown method
        self._get_some_device()
        self._fill_device_info_performance_values()
        self.log().info("Will use the following device "
                        "(guid: dtype, rating, BLOCK_SIZE, memalign):")
        for dtype in sorted(opencl_types.dtypes.keys()):
            self.log().info("%s: %s, %.2f, %d, %d" % (
                self.info.guid, dtype, self.info.rating[dtype],
                self.info.BLOCK_SIZE[dtype], self.info.memalign))

    def init_unpickled(self):
        super(Device, self).init_unpickled()
        self.context_ = None
        self.queue_ = None
        self.pid_ = os.getpid()

    def _get_some_device(self):
        """Gets some device from the available OpenCL devices.
        """
        self.context_ = pyopencl.create_some_context()
        if self.context_ == None:
            raise error.ErrNotExists("Could not create OpenCL context.")
        device = self.context_.devices[0]
        s = device.get_info(pyopencl.device_info.VERSION)
        n = s.find(" ") + 1
        m = s.find(" ", n)
        self.info = DeviceInfo(guid="%s/%s/%s" % (
            device.get_info(pyopencl.device_info.VENDOR).strip(),
            device.get_info(pyopencl.device_info.NAME).strip(),
            str(device.get_info(pyopencl.device_info.VENDOR_ID))),
            memsize=device.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE),
            memalign=device.get_info(pyopencl.device_info.MEM_BASE_ADDR_ALIGN),
            version=float(s[n:m]))
        self.queue_ = pyopencl.CommandQueue(self.context_,
            properties=pyopencl.command_queue_properties.\
            OUT_OF_ORDER_EXEC_MODE_ENABLE)

    def _fill_device_info_performance_values(self):
        device_infos = {}
        try:
            fin = open("%s/device_infos.pickle" % (config.cache_dir), "rb")
            device_infos = pickle.load(fin)
            fin.close()
        except IOError:
            self.log().info("%s/device_infos.pickle was not found" % (
                                                        config.cache_dir))
        if (not config.test_known_device and
            self.info.guid in device_infos.keys()):
            info = device_infos[self.info.guid]
            self.info.rating.update(info.rating)
            self.info.BLOCK_SIZE.update(info.BLOCK_SIZE)
            self.info.dt.update(info.dt)
            self.info.min_dt.update(info.min_dt)
            return
        if not config.test_unknown_device:
            return
        self._do_tests(device_infos)
        device_infos[self.info.guid] = self.info
        self.log().info("Saving found device performance values into "
                        "%s/device_infos.pickle" % (config.cache_dir))
        fout = open("%s/device_infos.pickle" % (config.cache_dir), "wb")
        pickle.dump(device_infos, fout)
        fout.close()
        self.log().info("Saved")

    def _do_tests(self, device_infos):
        """Measure relative device performance.
        """
        self.log().info("Will test device performance.\n"
            "Results of the test will be saved to %s/device_infos.pickle, "
            "so this is one time process usually." % (config.cache_dir))

        min_dt = {}
        for dtype in opencl_types.dtypes.keys():
            min_dt[dtype] = 86400
        dt_numpy = 86400
        for info in device_infos.values():
            for dtype in info.min_dt.keys():
                min_dt[dtype] = info.min_dt[dtype]
            break

        cc = {}
        for dtype in self.info.dt.keys():
            self.info.dt[dtype] = 86400
        for BLOCK_SIZE in range(32, 3, -1):
            for dtype in sorted(opencl_types.dtypes.keys()):
                try:
                    self._prepare_tests(BLOCK_SIZE, dtype)
                    key = "%s_%d_%d_%d" % (
                        "double2" if dtype[-1] == "2"
                        else "double", self.AB_WIDTH,
                        self.B_HEIGHT, self.A_HEIGHT)
                    if not key in cc.keys():
                        self.log().info("Numpy double precision "
                                        "for dtype=%s" % (dtype))
                        dt = self._do_cpu_test(cc, key)
                        self.log().info("Done in %.2f seconds" % (dt))
                        if dt < dt_numpy:
                            dt_numpy = dt
                        if dt_numpy < min_dt[dtype]:
                            min_dt[dtype] = dt_numpy
                    self.log().info("Testing %s with BLOCK_SIZE = %d "
                        "and dtype = %s" % (self.info.guid, BLOCK_SIZE, dtype))
                    dt = self._do_test(BLOCK_SIZE, dtype, 3)
                    if dt < self.info.dt[dtype]:
                        self.info.dt[dtype] = dt
                        self.info.BLOCK_SIZE[dtype] = BLOCK_SIZE
                    if dt < min_dt[dtype]:
                        min_dt[dtype] = dt
                    c = cc[key].copy()
                    c -= self.c.v
                    c = numpy.sqrt(numpy.square(numpy.real(c)) +
                                   numpy.square(numpy.imag(c)))
                    self.log().info("Avg is %.2f seconds, MSE = %.6f, "
                                    "max_diff = %.6f" % (
                                    dt, numpy.sum(c) / c.size, c.max()))
                    self._cleanup_after_tests()
                except (pyopencl.LogicError, pyopencl.RuntimeError,
                        pyopencl.MemoryError):
                    a, b, c = sys.exc_info()
                    self.log().info("Program compilation or run failed for "
                        "BLOCK_SIZE = %d and dtype = %s "
                        "(details in stderr)" % (BLOCK_SIZE, dtype))
                    traceback.print_exception(a, b, c)
                    self._cleanup_after_tests()

        del cc

        self.log().info("\nRating(numpy double precision): %.4f" % (
            min_dt[config.dtype] / dt_numpy))
        for info in device_infos.values():
            for dtype in sorted(opencl_types.dtypes.keys()):
                self.log().info("================")
                self.log().info(dtype)
                rating = min_dt[dtype] / info.dt[dtype]
                if info.rating[dtype] != rating:
                    if info.rating[dtype]:
                        self.log().info("UPD Rating(%s): %.4f" % (info.guid,
                                                                  rating))
                    else:
                        self.log().info("NEW Rating(%s): %.4f" % (info.guid,
                                                                  rating))
                else:
                    self.log().info("Rating(%s): %.4f" % (info.guid, rating))
                info.rating[dtype] = rating
                info.min_dt[dtype] = min_dt[dtype]
        self.log().info("================")

    def _prepare_tests(self, BLOCK_SIZE, dtype):
        self.AB_WIDTH = 65537
        self.B_HEIGHT = 257
        self.A_HEIGHT = 511
        if False:
            self.AB_WIDTH = formats.roundup(self.AB_WIDTH, BLOCK_SIZE)
            self.B_HEIGHT = formats.roundup(self.B_HEIGHT, BLOCK_SIZE)
            self.A_HEIGHT = formats.roundup(self.A_HEIGHT, BLOCK_SIZE)
        self.log().info("Matricies are: [%d, %d] * [%d, %d] = [%d, %d]" % (
            self.AB_WIDTH, self.A_HEIGHT, self.B_HEIGHT, self.AB_WIDTH,
            self.A_HEIGHT, self.B_HEIGHT))
        self.rnd_state = rnd.default.state

        self.a = formats.Vector()
        self.a.v = numpy.zeros([self.A_HEIGHT, self.AB_WIDTH],
                               dtype=opencl_types.dtypes[dtype])
        rnd.default.fill(self.a.v, -0.1, 0.1)

        self.b = formats.Vector()
        self.b.v = numpy.zeros([self.B_HEIGHT, self.AB_WIDTH],
                               dtype=opencl_types.dtypes[dtype])
        rnd.default.fill(self.b.v, -0.1, 0.1)

        self.bias = formats.Vector()
        self.bias.v = numpy.zeros(self.B_HEIGHT,
                                  dtype=opencl_types.dtypes[dtype])
        rnd.default.fill(self.bias.v, -0.1, 0.1)

        self.c = formats.Vector()
        self.c.v = numpy.zeros([self.A_HEIGHT, self.B_HEIGHT],
                               dtype=opencl_types.dtypes[dtype])

    def _cleanup_after_tests(self):
        del(self.c)
        del(self.bias)
        del(self.b)
        del(self.a)
        rnd.default.state = self.rnd_state
        del(self.rnd_state)
        del(self.A_HEIGHT)
        del(self.B_HEIGHT)
        del(self.AB_WIDTH)

    def _do_cpu_test(self, cc, key):
        """Pure single core CPU test
        """
        dtype = (numpy.complex128 if self.a.v.dtype in (
                    numpy.complex64, numpy.complex128) else numpy.float64)
        a = numpy.empty(self.a.v.shape, dtype=dtype)
        a[:] = self.a.v[:]
        bt = self.b.v.transpose()
        b = numpy.empty(bt.shape, dtype=dtype)
        b[:] = bt[:]
        bias = numpy.empty(self.bias.v.shape, dtype=dtype)
        bias[:] = self.bias.v[:]
        c = numpy.empty(self.c.v.shape, dtype=dtype)
        t1 = time.time()
        numpy.dot(a, b, c)
        c[:] += bias
        c *= 0.6666
        numpy.tanh(c, c)
        c *= 1.7159
        dt = time.time() - t1
        cc[key] = c
        return dt

    def _do_test(self, BLOCK_SIZE, dtype, iters):
        """Do test for specific context
        """
        defines = ("%s\n"
        "#define ACTIVATION_TANH\n"
        "#define BLOCK_SIZE %d\n"
        "#define H %d\n"
        "#define Y %d\n"
        "#define BATCH %d\n\n" % (config.cl_defines[dtype], BLOCK_SIZE,
                                  self.AB_WIDTH, self.B_HEIGHT, self.A_HEIGHT))
        s = defines
        s += units.OpenCLUnit.read_ocl_file("defines.cl")
        s_mx_mul = units.OpenCLUnit.read_ocl_file("matrix_multiplication.cl")
        s += units.OpenCLUnit.read_ocl_file("forward.cl")
        s = s.replace("MX_MUL", s_mx_mul)
        fout = open(os.path.join(config.cache_dir, "test.cl"), "w")
        fout.write(s)
        fout.close()

        self.a.initialize(self)
        self.b.initialize(self)
        self.c.initialize(self)
        self.bias.initialize(self)

        prg = pyopencl.Program(self.context_, s).build()

        krn = pyopencl.Kernel(prg, "feed_layer")
        krn.set_arg(0, self.a.v_)
        krn.set_arg(1, self.b.v_)
        krn.set_arg(2, self.c.v_)
        krn.set_arg(3, self.bias.v_)

        global_size = [formats.roundup(self.B_HEIGHT, BLOCK_SIZE),
                       formats.roundup(self.A_HEIGHT, BLOCK_SIZE)]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]
        t1 = time.time()
        # Will skip the first iteration
        for i in range(0, iters + 1):
            if i == 1:
                t1 = time.time()
            event = pyopencl.enqueue_nd_range_kernel(self.queue_, krn,
                                            global_size, local_size)
            event.wait()
        dt = time.time() - t1
        # Get results back
        self.c.map_read()
        return dt / iters
