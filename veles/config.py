"""
Created on May 28, 2013

Global configuration variables.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

import veles.opencl_types as opencl_types


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

# : Global config
root = Config()
root.common = Config()


def get(value, default_value=None):
    """Gets value from global config.
    """
    if isinstance(value, Config):
        return default_value
    return value


root.common.update = {"graphics_multicast_address": "239.192.1.1",
                      "matplotlib_backend": "Qt4Agg",
                      "matplotlib_webagg_port": 8081,
                      "mongodb_logging_address": "smaug:27017",
                      "plotters_disabled": False,
                      "precision_type": "double",
                      "test_dataset_root": "/data/veles",
                      "test_known_device": False,
                      "test_unknown_device": True,
                      "unit_test": False,
                      "veles_dir":
                      os.path.dirname(os.path.dirname(__file__)),
                      "veles_user_dir":
                      os.path.join(os.environ.get("HOME", "./"), "velesuser"),
                      "web_status_host": "smaug",
                      "web_status_log_file":
                      "/var/log/veles/web_status.log",
                      "web_status_notification_interval": 1,
                      "web_status_port": 8090,
                      "web_status_update": "update",
                      "web_status_pidfile": "/var/run/veles/web_status.pid",
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

root.common.device_dir = os.path.join(root.common.veles_dir, "devices")
root.common.dtype = opencl_types.dtype_map[root.common.precision_type]
root.common.ocl_dirs = (os.environ.get("VELES_OPENCL_DIRS", "").split(":") +
                        [os.path.join(root.common.veles_dir, "ocl")])
root.common.opencl_dir = os.path.join(root.common.veles_dir, "veles")
root.common.test_dataset_root = "/data/veles"
root.common.web_status_root = os.path.join(root.common.veles_dir, "web_status")
