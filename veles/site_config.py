import os

from veles.paths import __root__


def update(root):
    root.common.update({
        "mongodb_logging_address": "smaug:27017",
        "test_dataset_root": "/data/veles",
        "help_dir": os.path.join(__root__, "docs/html"),
        "web": {
            "host": "smaug",
            "port": 8090,
        },
        "engine": {
            "dirs": (os.environ.get("VELES_ENGINE_DIRS", "").split(":") +
                     [__root__])
        }
    })

    root.common.device_dirs.append(os.path.join(__root__, "devices"))
    root.common.test_dataset_root = "/data/veles/datasets"
