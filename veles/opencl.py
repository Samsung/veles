"""
Created on Mar 21, 2013

OpenCL base classes.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
import json
import logging
import numpy
import os
from six import add_metaclass
import sys
import opencl4py as cl

from .cmdline import CommandLineArgumentsRegistry
from .compat import from_none
from .config import root
from .distributable import Pickleable
import veles.opencl_types as opencl_types
import veles.external.prettytable as prettytable


PYVER = sys.version_info[0]


class DeviceInfo(object):
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
        self.max_block_size = kwargs["max_block_size"]
        self.rating = {}
        self.device_info = {}

    def get_block_size(self, **kwargs):
        """Gets optimal block size for matrix multiplication.

        Parameters:
            dtype: numeric data type as string (float or double).
            kernel: hint for the name of the kernel for which the optimal
                    block sizes will be returned:
                    conv: convolutional forward propagation,
                    deconv: convolutional back propagation,
                    all other: simple matrix multiplication.
            precision: precision level for summation (0, 1, 2)
                       (defaults to root.common.precision_level).

        Returns:
            BLOCK_SIZE
        """
        dtype = kwargs["dtype"]
        if type(dtype) != str:
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        krnnme = kwargs.get("kernel", "matrix_multiplication")
        precision = kwargs.get("precision", root.common.precision_level)
        krninfo = self.device_info.get(krnnme)
        if krninfo is None:
            logging.warning(
                "krnnme = %s was not found, "
                "rolling back to block size for matrix_multiplication",
                krnnme)
            krnnme = "matrix_multiplication"
            krninfo = self.device_info.get(krnnme)
            if krninfo is None:
                logging.warning(
                    "krnnme = %s was not found, "
                    "rolling back to the default block size", krnnme)
                return self.default_block_size
        typeinfo = krninfo.get(dtype)
        if typeinfo is None:
            logging.warning(
                "dtype = %s was not found with krnnme = %s and "
                "rolling back to the default block size", dtype, krnnme)
            return self.default_block_size
        bs_dt = typeinfo.get(str(precision))
        while bs_dt is None and precision > 0:
            precision -= 1
            bs_dt = typeinfo.get(str(precision))
        if bs_dt is None:
            logging.warning(
                "precision = 0 was not found with krnnme = %s and dtype = %s, "
                "rolling back to the default block size", krnnme, dtype)
            return self.default_block_size
        return bs_dt[0]

    @property
    def default_block_size(self):
        return self.max_block_size

    @property
    def vector_opt(self):
        return self.device_type == cl.CL_DEVICE_TYPE_CPU


class DeviceNotFoundError(Exception):
    pass


@add_metaclass(CommandLineArgumentsRegistry)
class Device(Pickleable):
    """OpenCL device helper class.

    Attributes:
        device_info: DeviceInfo object.
        context_: OpenCL context handle.
        queue_: OpenCL device queue.
        pid_: process id.
    """
    def __init__(self):
        super(Device, self).__init__()

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
        log_configs = "Selected the following OpenCL configurations:\n"
        table = prettytable.PrettyTable("device", " dtype", "rating",
                                        "BLOCK_SIZE", "version")
        table.align["device"] = "l"
        table.align[" dtype"] = "l"
        table.align["BLOCK_SIZES"] = "l"
        for dtype in sorted(opencl_types.dtypes.keys()):
            rating = self.device_info.rating.get(dtype)
            if rating is None:
                rating = ""
            else:
                rating = "%.3f" % rating
            table.add_row(self.device_info.desc, dtype, rating,
                          self.device_info.get_block_size(dtype=dtype),
                          self.device_info.version)
        self.info(log_configs + str(table))

    @property
    def max_block_size(self):
        sz = int(numpy.sqrt(self.queue_.device.max_work_group_size))
        sh = self.queue_.device.max_work_item_sizes
        return min(sz, sh[0], sh[1])

    @property
    def exists(self):
        return self.queue_ is not None

    def init_unpickled(self):
        super(Device, self).init_unpickled()
        self.queue_ = None
        self.pid_ = os.getpid()

    @staticmethod
    def arg_completer(prefix, **kwargs):
        def format_device(platform, device):
            return "%s - %s on %s" % (device.path, device.name.strip(),
                                      platform.name)

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
        parser.add_argument(
            "-d", "--device", type=str, default="",
            help="OpenCL device to use.").completer = Device.arg_completer
        return parser

    @property
    def max_group_size(self):
        return self.queue_.device.max_work_group_size

    def _get_some_device(self, **kwargs):
        """Gets some device from the available OpenCL devices.
        Returns True if any device was selected, otherwise, False.
        """
        parser = Device.init_parser(**kwargs)
        args, _ = parser.parse_known_args()
        try:
            platforms = cl.Platforms()
        except cl.CLRuntimeError:
            platforms = None
        if platforms is None or len(platforms.platforms) == 0:
            self.warning("No OpenCL devices was found")
            return False
        if args.device == "":
            context = platforms.create_some_context()
        else:
            platfnum, devnums = args.device.split(':')
            try:
                platform = platforms.platforms[int(platfnum)]
            except IndexError:
                raise from_none(
                    DeviceNotFoundError("Device %s was not found." %
                                        args.device))
            context = platform.create_context(
                [platform.devices[int(devnum)]
                 for devnum in devnums.split(',')])
        device = context.devices[0]
        desc = "%s/%s/%d" % (device.vendor.strip(), device.name.strip(),
                             device.vendor_id)
        self.queue_ = context.create_queue(device)
        self.device_info = DeviceInfo(
            desc=desc, memsize=device.memsize,
            memalign=device.memalign, version=device.version,
            device_type=device.type, max_block_size=self.max_block_size)
        return True

    def _fill_device_info_performance_values(self):
        device_infos = {}
        try:
            with open(os.path.join(root.common.device_dir,
                                   "device_infos.json"), "r") as fin:
                device_infos = json.load(fin)
        except IOError:
            self.warning("%s was not found",
                         os.path.join(root.common.device_dir,
                                      "device_infos.json"))
        self.compute_rating(device_infos)
        if self.device_info.desc not in device_infos:
            self.warning("Device is not in a database, "
                         "will use the default block sizes")
            return
        self.device_info.device_info = device_infos[self.device_info.desc]

    def compute_rating(self, device_infos):
        devdt = {}
        min_dt = {}
        for desc, device_info in sorted(device_infos.items()):
            krninfo = device_info.get("matrix_multiplication")
            if krninfo is None:
                continue
            devdt[desc] = {}
            for dtype, typeinfo in krninfo.items():
                bsdt = typeinfo.get("0")
                if bsdt is None:
                    continue
                devdt[desc][dtype] = bsdt[1]
                min_dt[dtype] = min(min_dt.get(dtype, 1.0e30), bsdt[1])

        table = prettytable.PrettyTable("device", " dtype", "rating")
        table.align["device"] = "l"
        table.align[" dtype"] = "l"
        rating = {}
        for desc, dtypedt in sorted(devdt.items()):
            rating[desc] = {}
            for dtype, dt in sorted(dtypedt.items()):
                rating[desc][dtype] = min_dt[dtype] / dt
                table.add_row(self.device_info.desc, dtype,
                              "%.3f" % rating[desc][dtype])
        self.debug("Device ratings:\n%s", str(table))

        if self.device_info.desc in rating:
            self.device_info.rating = rating[self.device_info.desc]
