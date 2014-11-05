import os

from .paths import __root__


def update(root):
    root.common.update({
        "mongodb_logging_address": "smaug:27017",
        "test_dataset_root": "/data/veles",
        "device_dir": os.path.join(__root__, "devices"),
        "ocl_dirs": (os.environ.get("VELES_OPENCL_DIRS", "").split(":") +
                     [os.path.join(__root__, "ocl")]),
        "help_dir": os.path.join(__root__, "docs/html"),
        "web": {
            "host": "smaug",
            "port": 8090,
        },
    })

    root.common.test_dataset_root = "/data/veles/datasets"
