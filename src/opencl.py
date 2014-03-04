"""
Created on Mar 21, 2013

OpenCL helper classes.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import os
import pickle
import sys
import time
import traceback
import opencl4py as cl
import config
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
            self.BLOCK_SIZE[dtype] = 8


class Device(units.Pickleable):
    """OpenCL device helper class.

    Attributes:
        device_info: DeviceInfo object.
        context_: OpenCL context handle.
        queue_: OpenCL device queue.
        pid_: process id.
    """
    def __init__(self):
        super(Device, self).__init__()
        self._get_some_device()
        self._fill_device_info_performance_values()
        self.info("Will use the following device "
                  "(guid: dtype, rating, BLOCK_SIZE, memalign, version):")
        for dtype in sorted(opencl_types.dtypes.keys()):
            self.info("%s: %s, %.2f, %d, %d, %.1f" % (
                self.device_info.guid, dtype, self.device_info.rating[dtype],
                self.device_info.BLOCK_SIZE[dtype], self.device_info.memalign,
                self.device_info.version))

    def init_unpickled(self):
        super(Device, self).init_unpickled()
        self.queue_ = None
        self.pid_ = os.getpid()

    def _get_some_device(self):
        """Gets some device from the available OpenCL devices.
        """
        platforms = cl.Platforms()
        context = platforms.create_some_context()
        device = context.devices[0]
        guid = "%s/%s/%d" % (device.vendor.strip(), device.name.strip(),
                             device.vendor_id)
        self.device_info = DeviceInfo(guid=guid, memsize=device.memsize,
            memalign=device.memalign, version=device.version)
        self.queue_ = context.create_queue(device,
            cl.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)

    def _fill_device_info_performance_values(self):
        device_infos = {}
        try:
            fin = open("%s/device_infos.pickle" % (config.cache_dir), "rb")
            device_infos = pickle.load(fin)
            fin.close()
        except IOError:
            self.info("%s/device_infos.pickle was not found" % (
                                                        config.cache_dir))
        if (not config.test_known_device and
            self.device_info.guid in device_infos.keys()):
            device_info = device_infos[self.device_info.guid]
            self.device_info.rating.update(device_info.rating)
            self.device_info.BLOCK_SIZE.update(device_info.BLOCK_SIZE)
            self.device_info.dt.update(device_info.dt)
            self.device_info.min_dt.update(device_info.min_dt)
            return
        if not config.test_unknown_device:
            return
        device_infos[self.device_info.guid] = self.device_info
        self._do_tests(device_infos)
        self.info("Saving found device performance values into "
                        "%s/device_infos.pickle" % (config.cache_dir))
        fout = open("%s/device_infos.pickle" % (config.cache_dir), "wb")
        pickle.dump(device_infos, fout)
        fout.close()
        self.info("Saved")

    def _do_tests(self, device_infos):
        """Measure relative device performance.
        """
        self.info("Will test device performance.\n"
            "Results of the test will be saved to %s/device_infos.pickle, "
            "so this is one time process usually." % (config.cache_dir))

        min_dt = {}
        dt_numpy = {}
        for dtype in opencl_types.dtypes.keys():
            min_dt[dtype] = 86400
            dt_numpy[dtype] = 86400
        for device_info in device_infos.values():
            for dtype in device_info.min_dt.keys():
                min_dt[dtype] = device_info.min_dt[dtype]
            break

        cc = {}
        for dtype in self.device_info.dt.keys():
            self.device_info.dt[dtype] = 86400
        for BLOCK_SIZE in range(32, 3, -1):
            for dtype in sorted(opencl_types.dtypes.keys()):
                try:
                    self._prepare_test(BLOCK_SIZE, dtype, cc)
                    key = "%s_%d_%d_%d" % (dtype, self.AB_WIDTH,
                        self.B_HEIGHT, self.A_HEIGHT)
                    if not key in cc.keys():
                        self.info("Numpy for dtype=%s" % (dtype))
                        dt = self._do_cpu_test(cc, key)
                        self.info("Done in %.3f seconds" % (dt))
                        if dt < dt_numpy[dtype]:
                            dt_numpy[dtype] = dt
                        if dt_numpy[dtype] < min_dt[dtype]:
                            min_dt[dtype] = dt_numpy[dtype]
                    self.info("Testing %s with BLOCK_SIZE = %d "
                        "and dtype = %s" % (self.device_info.guid, BLOCK_SIZE,
                                            dtype))
                    dt = self._do_test(BLOCK_SIZE, dtype, 3)
                    if dt < self.device_info.dt[dtype]:
                        self.device_info.dt[dtype] = dt
                        self.device_info.BLOCK_SIZE[dtype] = BLOCK_SIZE
                    if dt < min_dt[dtype]:
                        min_dt[dtype] = dt
                    key = "%s_%d_%d_%d" % ("double2" if dtype[-1] == "2"
                        else "double", self.AB_WIDTH,
                        self.B_HEIGHT, self.A_HEIGHT)
                    c = cc[key].copy()
                    c -= self.c.v
                    c = numpy.sqrt(numpy.square(numpy.real(c)) +
                                   numpy.square(numpy.imag(c)))
                    self.info("Avg is %.3f seconds, MSE = %.6f, "
                                    "max_diff = %.6f" % (
                                    dt, numpy.sum(c) / c.size, c.max()))
                    self._cleanup_after_test()
                except RuntimeError:
                    a, b, c = sys.exc_info()
                    self.info("Program compilation or run failed for "
                        "BLOCK_SIZE = %d and dtype = %s "
                        "(details in stderr)" % (BLOCK_SIZE, dtype))
                    traceback.print_exception(a, b, c)
                    self._cleanup_after_test()

        del cc

        for dtype in sorted(opencl_types.dtypes.keys()):
            self.info("Rating of numpy for dtype = %s: %.4f" % (
                dtype, min_dt[dtype] / dt_numpy[dtype]))
        for device_info in device_infos.values():
            for dtype in sorted(opencl_types.dtypes.keys()):
                self.info("================")
                self.info(dtype)
                rating = min_dt[dtype] / device_info.dt[dtype]
                if device_info.rating[dtype] != rating:
                    if device_info.rating[dtype]:
                        self.info("UPD Rating(%s): %.4f" % (device_info.guid,
                                                                  rating))
                    else:
                        self.info("NEW Rating(%s): %.4f" % (device_info.guid,
                                                                  rating))
                else:
                    self.info("Rating(%s): %.4f" % (device_info.guid, rating))
                device_info.rating[dtype] = rating
                device_info.min_dt[dtype] = min_dt[dtype]
        self.info("================")

    def _prepare_test(self, BLOCK_SIZE, dtype, cc):
        self.AB_WIDTH = 3001
        self.B_HEIGHT = 3001
        self.A_HEIGHT = 3001
        #self.AB_WIDTH = formats.roundup(self.AB_WIDTH, BLOCK_SIZE)
        #self.B_HEIGHT = formats.roundup(self.B_HEIGHT, BLOCK_SIZE)
        #self.A_HEIGHT = formats.roundup(self.A_HEIGHT, BLOCK_SIZE)
        self.info("Matricies are: [%d, %d] * [%d, %d] = [%d, %d]" % (
            self.AB_WIDTH, self.A_HEIGHT, self.B_HEIGHT, self.AB_WIDTH,
            self.A_HEIGHT, self.B_HEIGHT))
        self.rnd_state = rnd.default.state

        xdtype = ("complex" if dtype in (numpy.complex64, numpy.complex128)
                  else "real")

        self.a = formats.Vector()
        self.a.v = numpy.zeros([self.A_HEIGHT, self.AB_WIDTH],
                               dtype=opencl_types.dtypes[dtype])
        a_rnd = cc.get("a_rnd")
        if a_rnd == None:
            a_rnd = {}
            cc["a_rnd"] = a_rnd
        if a_rnd.get(xdtype) == None:
            rnd.default.fill(self.a.v, -0.1, 0.1)
            a_rnd[xdtype] = self.a.v.copy()
        else:
            self.a.v[:] = a_rnd[xdtype][:]

        self.b = formats.Vector()
        self.b.v = numpy.zeros([self.B_HEIGHT, self.AB_WIDTH],
                               dtype=opencl_types.dtypes[dtype])
        b_rnd = cc.get("b_rnd")
        if b_rnd == None:
            b_rnd = {}
            cc["b_rnd"] = b_rnd
        if b_rnd.get(xdtype) == None:
            rnd.default.fill(self.b.v, -0.1, 0.1)
            b_rnd[xdtype] = self.b.v.copy()
        else:
            self.b.v[:] = b_rnd[xdtype][:]

        self.bias = formats.Vector()
        self.bias.v = numpy.zeros(self.B_HEIGHT,
                                  dtype=opencl_types.dtypes[dtype])
        bias_rnd = cc.get("bias_rnd")
        if bias_rnd == None:
            bias_rnd = {}
            cc["bias_rnd"] = bias_rnd
        if bias_rnd.get(xdtype) == None:
            rnd.default.fill(self.bias.v, -0.1, 0.1)
            bias_rnd[xdtype] = self.bias.v.copy()
        else:
            self.bias.v[:] = bias_rnd[xdtype][:]

        self.c = formats.Vector()
        self.c.v = numpy.zeros([self.A_HEIGHT, self.B_HEIGHT],
                               dtype=opencl_types.dtypes[dtype])

    def _cleanup_after_test(self):
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
        """Pure single core CPU test.
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
        """Do test for specific context.
        """
        obj = units.OpenCLUnit(None, device=self)
        obj.cl_sources_["forward.cl"] = {}
        defines = {
            "ACTIVATION_TANH": 1,
            "BLOCK_SIZE": BLOCK_SIZE,
            "H": self.AB_WIDTH,
            "Y": self.B_HEIGHT,
            "BATCH": self.A_HEIGHT}
        obj.build_program(defines, os.path.join(config.cache_dir, "test.cl"),
                          dtype=dtype)

        krn = obj.get_kernel("feed_layer")

        self.a.initialize(self)
        self.b.initialize(self)
        self.c.initialize(self)
        self.bias.initialize(self)

        krn.set_arg(0, self.a.v_)
        krn.set_arg(1, self.b.v_)
        krn.set_arg(2, self.c.v_)
        krn.set_arg(3, self.bias.v_)

        global_size = [formats.roundup(self.B_HEIGHT, BLOCK_SIZE),
                       formats.roundup(self.A_HEIGHT, BLOCK_SIZE)]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]
        t1 = time.time()
        # Will skip the first iteration
        ev = None
        for i in range(iters + 1):
            ev = self.queue_.execute_kernel(krn, global_size, local_size,
                wait_for=(None if ev == None else (ev,)))
            if i == 0:
                self.queue_.flush()
                ev.wait()
                ev = None
                t1 = time.time()
        self.queue_.flush()
        ev.wait()
        dt = time.time() - t1
        # Get results back
        self.c.map_read()
        return dt / iters
