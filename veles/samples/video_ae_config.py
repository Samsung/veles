#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os

from veles.config import root, Config


root.decision = Config()  # not necessary for execution (it will do it in real
# time any way) but good for Eclipse editor

# optional parameters

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "video_ae"},
               "global_alpha": 0.0002,
               "global_lambda": 0.00005,
               "layers": [9, 14400],
               "loader": {"minibatch_maxsize": 50},
               "path_for_load_data":
               os.path.join(root.common.test_dataset_root,
                            "video/video_ae/img/*.png"),
               "weights_plotter": {"limit": 16}
               }
