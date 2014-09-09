"""
Created on May 28, 2013

Global configuration variables.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import pprint


class Config(object):
    """Config service class.
    """

    def _update(self, tree):
        for k, v in tree.items():
            if isinstance(v, dict):
                getattr(self, k)._update(v)
            else:
                setattr(self, k, v)

    def _defaults(self, tree):
        for k, v in tree.items():
            if isinstance(v, dict):
                getattr(self, k)._defaults(v)
            else:
                attr = getattr(self, k)
                if isinstance(attr, Config):
                    setattr(self, k, v)

    def __getattr__(self, name):
        temp = Config()
        self.__setattr__(name, temp)
        return temp

    def __setattr__(self, name, value):
        if name == "update":
            if isinstance(value, dict):
                self._update(value)
                return
        if name == "defaults":
            if isinstance(value, dict):
                self._defaults(value)
                return
        super(Config, self).__setattr__(name, value)

    def __repr__(self):
        return str(self.__dict__)

    def print_config(self, indent=1, width=80):
        print('-' * width)
        print("Current configuration:")

        def fix_contents(contents):
            fixed_contents = dict(contents)
            for k, v in contents.items():
                if isinstance(v, Config):
                    fixed_contents[k] = fix_contents(v.__dict__)
            return fixed_contents

        contents = fix_contents(self.__dict__)
        pprint.pprint(contents, indent=1, width=width)
        print('-' * width)

# : Global config
root = Config()
root.common = Config()


def get(value, default_value=None):
    """Gets value from global config.
    """
    if isinstance(value, Config):
        return default_value
    return value


__path__ = os.path.dirname(os.path.dirname(__file__))

root.common.update = {
    "graphics_multicast_address": "239.192.1.1",
    "matplotlib_backend": "Qt4Agg",
    "matplotlib_webagg_port": 8081,
    "mongodb_logging_address": "smaug:27017",
    "plotters_disabled": False,
    "precision_type": "double",  # float or double
    "precision_level": 1,  # 0 - use simple summation
                           # 1 - use Kahan summation (9% slower)
                           # 2 - use multipartials summation (90% slower)
    "test_dataset_root": "/data/veles",
    "test_known_device": False,
    "test_unknown_device": True,
    "unit_test": False,
    "veles_dir": __path__,
    "veles_user_dir": os.path.join(os.environ.get("HOME", "./"), "velesuser"),
    "device_dir": os.path.join(__path__, "devices"),
    "ocl_dirs": (os.environ.get("VELES_OPENCL_DIRS", "").split(":") +
                 [os.path.join(__path__, "ocl")]),
    "opencl_dir": os.path.join(__path__, "veles"),
    "web": {
        "host": "smaug",
        "port": 8090,
        "log_file": "/var/log/veles/web_status.log",
        "log_backups": 9,
        "notification_interval": 1,
        "pidfile": "/var/run/veles/web_status",
        "root": os.path.join(__path__, "web"),
    },
    "ThreadPool": {
        "minthreads": 2,
        "maxthreads": 2,
    }
}

root.common.cache_dir = os.path.join(root.common.veles_user_dir, "cache")

try:
    os.makedirs(root.common.cache_dir)
except OSError:
    pass

root.common.snapshot_dir = os.path.join(root.common.veles_user_dir,
                                        "snapshots")
try:
    os.makedirs(root.common.snapshot_dir)
except OSError:
    pass

root.common.test_dataset_root = "/data/veles/datasets"
