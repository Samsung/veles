"""
Created on May 28, 2013

Global configuration variables.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from collections import defaultdict
import os
import platform
from pprint import pprint
from six import print_
import sys

from veles.paths import __root__

# : Global config
root = None
__protected__ = defaultdict(set)


class Config(object):
    """Config service class.
    """
    def __init__(self, path):
        self.__path__ = path

    def __del__(self):
        if __protected__ is not None and self in __protected__:
            del __protected__[self]

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

    def protect(self, *names):
        __protected__[self].update(names)

    def __getattr__(self, name):
        temp = Config("%s.%s" % (self.__path__, name))
        setattr(self, name, temp)
        return temp

    def __setattr__(self, name, value):
        if name in __protected__[self]:
            raise AttributeError(
                "Attempted to change the protected configuration setting %s.%s"
                % (self.__path__, name))
        super(Config, self).__setattr__(name, value)

    @property
    def __content__(self):
        attrs = dict(self.__dict__)
        del attrs["__path__"]
        return attrs

    def __repr__(self):
        return '<Config "%s": %s>' % (self.__path__, repr(self.__content__))

    def print_(self, indent=1, width=80, file=sys.stdout):
        print_('-' * width, file=file)
        print_('Configuration "%s":' % self.__path__, file=file)
        pprint(self.__getstate__(), indent=indent, width=width, stream=file)
        print_('-' * width, file=file)

    def __getstate__(self):
        def fix_contents(contents):
            fixed_contents = dict(contents)
            del fixed_contents["__path__"]
            for k, v in contents.items():
                if isinstance(v, Config):
                    fixed_contents[k] = fix_contents(v.__dict__)
            return fixed_contents

        return fix_contents(self.__dict__)

    def __setstate__(self, state):
        self.__update__(state)


root = Config("root")
# Preload "common"
root.common


def get(value, default_value=None):
    """Gets value from global config.
    """
    if isinstance(value, Config):
        return default_value
    return value


def validate_kwargs(caller, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, Config):
            caller.warning("Argument '%s' seems to be undefined at %s",
                           k, v.__path__)


__home__ = os.path.join(os.environ.get("HOME", "./"), ".veles")

root.common.update({
    "allow_root": False,
    "graphics_multicast_address": "239.192.1.1",
    "graphics_blacklisted_ifaces": set(),
    "matplotlib_backend": "Qt4Agg",
    "matplotlib_webagg_port": 8081,
    "mongodb_logging_address": "127.0.0.1:27017",
    "plotters_disabled": "unittest" not in sys.modules,
    "precision_type": "double",  # float or double
    "precision_level": 0,  # 0 - use simple summation
                           # Only for ocl backend:
                           # 1 - use Kahan summation (9% slower)
                           # 2 - use multipartials summation (90% slower)
    "pickles_compression": (None if
                            platform.python_implementation() == "CPython"
                            else "snappy"),
    "test_dataset_root": os.path.join(os.environ.get("HOME", "./"), "data"),
    "test_known_device": False,
    "test_unknown_device": True,
    "prefer_numpy_on_cpu": True,
    "disable_snapshots": False,
    "veles_dir": __root__,
    "veles_user_dir": __home__,
    "device_dirs": ["/usr/share/veles/devices",
                    os.path.join(__home__, "devices"),
                    os.environ.get("VELES_OPENCL_DEVICES", "./")],
    "help_dir": "/usr/share/doc/python3-veles",
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "log_file": "/var/log/veles/web_status.log",
        "log_backups": 9,
        "notification_interval": 1,
        "pidfile": "/var/run/veles/web_status",
        "root": os.path.join(__root__, "web"),
        "drop_time": 30 * 24 * 3600,
    },
    "forge": {
        "service_name": "service",
        "upload_name": "upload",
        "fetch_name": "fetch",
        "manifest": "manifest.json",
    },
    "ThreadPool": {
        "minthreads": 2,
        "maxthreads": 2,
    },
    "engine": {
        "backend": "ocl",
        "dirs": (os.environ.get("VELES_ENGINE_DIRS", "").split(":") +
                 ["/usr/share/veles"])
    }
})

root.common.web.templates = os.path.join(root.common.web.root, "templates")

# Allow to override the settings above
try:
    from veles.site_config import update
    update(root)
    del update
except ImportError:
    pass

root.common.cache_dir = os.path.join(root.common.veles_user_dir, "cache")
if not os.path.exists(root.common.cache_dir):
    os.makedirs(root.common.cache_dir)

root.common.snapshot_dir = os.path.join(root.common.veles_user_dir,
                                        "snapshots")
if not os.path.exists(root.common.snapshot_dir):
    os.makedirs(root.common.snapshot_dir)

if not root.common.allow_root and os.getuid() == 0:
    raise PermissionError(
        "I have detected your attempt to run this VELES-based script with root"
        " privileges. Most likely this is because the code must be fixed and "
        "you are too lazy to find out where and how. Bad, bad boy! "
        "If you REALLY need it, set root.common.allow_root to True.")

# Make some settings read-only
root.common.protect("cache_dir", "snapshot_dir", "pickles_compression",
                    "veles_user_dir")
