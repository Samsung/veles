#!/usr/bin/python3.3 -O
"""
Created on Mart 26, 2014

Example of Mnist config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os
from veles.config import root, Config

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters
root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "hands"},
               "global_alpha": 0.05,
               "global_lambda": 0.0,
               "layers_mnist": [30, 2],
               "loader": {"minibatch_maxsize": 60},
               "path_for_train_data":
               [os.path.join(root.common.test_dataset_root,
                             "hands/Positive/Training/*.raw"),
                os.path.join(root.common.test_dataset_root,
                             "hands/Negative/Training/*.raw")],
               "path_for_valid_data":
               [os.path.join(root.common.test_dataset_root,
                             "hands/Positive/Testing/*.raw"),
                os.path.join(root.common.test_dataset_root,
                             "hands/Negative/Testing/*.raw")]}
