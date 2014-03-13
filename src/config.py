"""
Created on May 28, 2013

Global configuration variables.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import os
import opencl_types


global sconfig
global _sconfig_empty


class Config(object):
    """Config service class.
    """
    def __getattr__(self, name):
        return _sconfig_empty


# : Global config
sconfig = Config()

# : Default config value
_sconfig_empty = Config()


def getConfig(value, default_value=None):
    """Gets value from global config.
    """
    if(value == _sconfig_empty):
        return default_value
    return value


# : Complex type
# c_dtype = "float"  # for real single precision numbers
c_dtype = "double"  # for real numbers
# c_dtype = "double2"  # for complex numbers
# c_dtype = "double4"  # for quad numbers (not implemented)

# : Real type
dtype = opencl_types.dtype_map[c_dtype]


# Directories

# : Directory with config.py itself
this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."

# : Directory for cache
cache_dir = os.path.join(this_dir, "../cache")
try:
    os.mkdir(cache_dir)
except OSError:
    pass

# : Directory for OpenCL source files
ocl_dirs = os.environ.get("VELES_OPENCL_DIRS", "").split(":") + \
           [os.path.join(this_dir, "../ocl")]

# : Directory where to save snapshots
snapshot_dir = os.path.join(this_dir, "../snapshots")
try:
    os.mkdir(snapshot_dir)
except OSError:
    pass

# : Directory where functional tests large datasets reside
test_dataset_root = "/data/veles"


"""
Globally disables all the plotting stuff.
"""
plotters_disabled = False


# : Test opencl device for optimal BLOCK_SIZE or not
test_known_device = False
test_unknown_device = True


# : For debugging purposes
unit_test = False

web_status_root = os.path.join(this_dir, "../web_status")
web_status_host = "smaug"
web_status_update = "update"
web_status_port = 8090
web_status_notification_interval = 1
web_status_log_file = "/var/log/veles/web_status.log"

graphics_multicast_address = "239.192.1.1"
matplotlib_backend = "Qt4Agg"
matplotlib_webagg_port = 8081

slaves = ["markovtsevu64", "seresovu64", "seninu64", "kuznetsovu64",
          "kazantsevu64", "lpodoynitsinau64", "smaug", "smaug",
          "smaug", "smaug"]
