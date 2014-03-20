#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Channels config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os

from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.update = {"cache_fnme":
               os.path.join(root.common.cache_dir, "channels.pickle"),
               "decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "channels_108_24",
                            "use_dynamic_alpha": False},
               "export": False,
               "find_negative": 0,
               "global_alpha": 0.01,
               "global_lambda": 0.00005,
               "layers": [108, 24],
               "grayscale": False,
               "loader": {"minibatch_maxsize": 81,
                          "rect": (264, 129)},
               "n_threads": 32,
               "path_for_train_data":
               "/data/veles/channels/korean_960_540/train",
               "snapshot": "",
               "validation_procent": 0.15,
               "weights_plotter.limit": 16}
