# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 28, 2013

Global configuration variables.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""

from collections import defaultdict
import os
import platform
from pprint import pprint
from six import print_, PY2
import sys

veles = __import__("veles")
from veles.paths import __root__, __home__

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
        if not isinstance(value, (dict, Config)):
            raise ValueError("Value must be an instance of dict or Config")
        self.__update__(
            value if isinstance(value, dict) else value.__content__)
        return self

    def __update__(self, tree):
        for k, v in tree.items():
            if isinstance(v, dict):
                getattr(self, k).__update__(v)
            else:
                setattr(self, k, v)

    def protect(self, *names):
        __protected__[self].update(names)

    def __getattr__(self, name):
        if name in ("__copy__", "__deepcopy__",):
            raise AttributeError()
        if name in ("keys", "values"):
            return getattr(self.__content__, name)
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
        if "__path__" in attrs:
            del attrs["__path__"]
        return attrs

    def __repr__(self):
        return '<Config "%s": %s>' % (self.__path__, repr(self.__content__))

    def print_(self, indent=1, width=80, file=sys.stdout):
        def fix_contents(obj):
            fixed_contents = content = obj.__content__
            for k, v in content.items():
                if isinstance(v, Config):
                    fixed_contents[k] = fix_contents(v)
            return fixed_contents

        print_('-' * width, file=file)
        print_('Configuration "%s":' % self.__path__, file=file)
        pprint(fix_contents(self), indent=indent, width=width, stream=file)
        print_('-' * width, file=file)

    def __getstate__(self):
        """
        Do not remove this method, if you think the default one works the same.
        It actually raises "Config object is not callable" exception.
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Do not remove this method, if you think the default one works the same.
        It actually leads to "maximum recursion depth exceeded" exception.
        """
        self.__dict__.update(state)

    def __iter__(self):
        return iter(self.__content__)

    def __getitem__(self, item):
        return getattr(self, item)

    if PY2:
        def __getnewargs__(self):
            return tuple()


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
        if isinstance(v, Config) and len(v.__content__) == 0:
            caller.warning("Argument '%s' seems to be undefined at %s",
                           k, v.__path__)
            if root.common.trace.undefined_configs:
                import inspect
                from traceback import format_list, extract_stack
                caller.warning("kwargs are: %s", kwargs)
                caller.warning("Stack trace:\n%s" %
                               "".join(format_list(extract_stack(
                                   inspect.currentframe().f_back))))

root.common.update({
    "mongodb_logging_address": "127.0.0.1:27017",
    "precision_type": "double",  # float or double
    "precision_level": 0,  # 0 - use simple summation
                           # Only for ocl backend:
                           # 1 - use Kahan summation (9% slower)
                           # 2 - use multipartials summation (90% slower)
    "datasets_root": os.path.join(__home__, "data"),
    "veles_dir": os.path.join(__root__, "veles"),
    "veles_user_dir": __home__,
    "veles_dist_config_dir": "/etc/default/veles",
    "help_dir": "/usr/share/doc/python3-veles",
    "disable": {
        "spinning_run_progress": not sys.stdout.isatty(),
        "plotting": "unittest" in sys.modules,
        "snapshotting": False,
        "publishing": False,
    },
    "trace": {
        "misprints": False,
        "undefined_configs": False,
        "run": False,
    },
    "warnings": {
        "numba": True
    },
    "exceptions": {
        "run_after_stop": False,
    },
    "graphics": {
        "multicast_address": "239.192.1.1",
        "blacklisted_ifaces": set(),
        "matplotlib": {
            "backend": "Qt4Agg",
            "webagg_port": 8081,
        }
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "log_file": "/var/log/veles/web_status.log",
        "log_backups": 9,
        "notification_interval": 1,
        "pidfile": "/var/run/veles/web_status",
        "root": "/usr/share/veles/web",
        "drop_time": 30 * 24 * 3600,
    },
    "api": {
        "port": 8180,
        "path": "/api"
    },
    "forge": {
        "service_name": "service",
        "upload_name": "upload",
        "fetch_name": "fetch",
        "manifest": "manifest.json",
        "tokens_file": os.path.join(__home__, "forge_tokens.txt"),
        "pending_file": os.path.join(__home__, "forge_pending.txt"),
        "email_templates": os.path.join(os.path.join(__root__, "veles",
                                                     "forge", "templates"))
    },
    "engine": {
        "backend": "auto",
        "test_known_device": False,
        "test_unknown_device": True,
        "test_precision_types": ("float", "double"),
        "test_precision_levels": (0, 1),
        "thread_pool": {
            "minthreads": 2,
            "maxthreads": 2,
        },
        # The following is a hack to make Intel OpenCL usable;
        # It does not have 64-bit atomics and the engine uses them
        "force_numpy_run_on_intel_opencl": True,
        # Disable Numba JIT while debugging or on alternative interpreters
        "disable_numba": (sys.gettrace() is not None or
                          platform.python_implementation() != "CPython"),
        "network_compression": (None if  # snappy is slow on CPython
                                platform.python_implementation() == "CPython"
                                else "snappy"),
        "source_dirs": (os.environ.get("VELES_ENGINE_DIRS", "").split(":") +
                        ["/usr/share/veles"]),
        "device_dirs": ["/usr/share/veles/devices",
                        os.path.join(__home__, "devices"),
                        os.environ.get("VELES_DEVICE_DIRS", "./")],
        "ocl": {
            # Use clBLAS if it is available
            "clBLAS": False
        },
        "cuda": {
            # Path to nvcc
            "nvcc": "nvcc"
        }
    },
    "genetics": {
        "disable": {
            "plotters": True
        },
        "evaluation_transform": lambda v, t: v
    }
})

# Allow to override the settings above
try:
    from veles.site_config import update
    update(root)
    del update
except ImportError:
    pass
for site_path in (root.common.veles_dist_config_dir,
                  root.common.veles_user_dir, os.getcwd()):
    sys.path.insert(0, site_path)
    try:
        __import__("site_config").update(root)
    except (ImportError, AttributeError):
        pass
    finally:
        del sys.path[0]

root.common.web.templates = os.path.join(root.common.web.root, "templates")
root.common.cache_dir = os.path.join(root.common.veles_user_dir, "cache")
root.common.snapshot_dir = os.path.join(root.common.veles_user_dir,
                                        "snapshots")
for d in (root.common.cache_dir, root.common.snapshot_dir,
          root.common.datasets_root):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError:
            pass

# Make some settings read-only
root.common.protect("cache_dir", "snapshot_dir", "pickles_compression",
                    "veles_user_dir")

if os.getuid() == 0 and not getattr(veles, "allow_root", False) \
        and os.getenv("VELES_ALLOW_ROOT") is None:
    raise PermissionError(
        "I have detected your attempt to run this VELES-based script with root"
        " privileges. Most likely this is because the code must be fixed and "
        "you are too lazy to find out where and how. Bad, bad boy! "
        "If you REALLY need it, set veles.allow_root to True or define "
        "VELES_ALLOW_ROOT environment variable.")
