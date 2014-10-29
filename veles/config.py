"""
Created on May 28, 2013

Global configuration variables.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import pprint

import veles

# : Global config
root = None


class Config(object):
    """Config service class.
    """

    def update(self, value):
        if self == root:
            raise ValueError("Root updates are disabled")
        if not isinstance(value, dict):
            raise ValueError("Value must be an instance of dict")
        self.__update__(value)

    def __update__(self, tree):
        for k, v in tree.items():
            if isinstance(v, dict):
                getattr(self, k).__update__(v)
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        temp = Config()
        setattr(self, name, temp)
        return temp

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)

    def __repr__(self):
        return str(self.__dict__)

    def print_config(self, indent=1, width=80):
        print('-' * width)
        print("Current configuration:")

        pprint.pprint(self.__getstate__(), indent=indent, width=width)
        print('-' * width)

    def __getstate__(self):
        def fix_contents(contents):
            fixed_contents = dict(contents)
            for k, v in contents.items():
                if isinstance(v, Config):
                    fixed_contents[k] = fix_contents(v.__dict__)
            return fixed_contents

        return fix_contents(self.__dict__)

    def __setstate__(self, state):
        self.__update__(state)


root = Config()
root.common = Config()


def get(value, default_value=None):
    """Gets value from global config.
    """
    if isinstance(value, Config):
        return default_value
    return value


__path__ = veles.__root__

root.common.update({
    "graphics_multicast_address": "239.192.1.1",
    "matplotlib_backend": "Qt4Agg",
    "matplotlib_webagg_port": 8081,
    "mongodb_logging_address": "0.0.0.0:27017",
    "plotters_disabled": False,
    "precision_type": "double",  # float or double
    "precision_level": 1,  # 0 - use simple summation
                           # 1 - use Kahan summation (9% slower)
                           # 2 - use multipartials summation (90% slower)
    "test_dataset_root": os.path.join(os.environ.get("HOME", "./"), "data"),
    "test_known_device": False,
    "test_unknown_device": True,
    "disable_snapshots": False,
    "unit_test": False,
    "veles_dir": __path__,
    "veles_user_dir": os.path.join(os.environ.get("HOME", "./"), ".veles"),
    "device_dir": "/usr/share/veles/devices",
    "ocl_dirs": (os.environ.get("VELES_OPENCL_DIRS", "").split(":") +
                 ["/usr/share/veles/ocl"]),
    "help_dir": "/usr/share/doc/veles/html",
    "web": {
        "host": "0.0.0.0",
        "port": 80,
        "log_file": "/var/log/veles/web_status.log",
        "log_backups": 9,
        "notification_interval": 1,
        "pidfile": "/var/run/veles/web_status",
        "root": os.path.join(__path__, "web"),
        "drop_time": 30 * 24 * 3600,
    },
    "ThreadPool": {
        "minthreads": 2,
        "maxthreads": 2,
    }
})

# Allow to override the settings above
try:
    import veles.siteconfig
except ImportError:
    pass

root.common.cache_dir = os.path.join(root.common.veles_user_dir, "cache")
if not os.path.exists(root.common.cache_dir):
    os.makedirs(root.common.cache_dir)

root.common.snapshot_dir = os.path.join(root.common.veles_user_dir,
                                        "snapshots")
if not os.path.exists(root.common.snapshot_dir):
    os.makedirs(root.common.snapshot_dir)
