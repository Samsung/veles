#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os

from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters
root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "mnist_784"},
               "global_alpha": 0.001,
               "global_lambda": 0.00005,
               "layers_mnist784": [784, 784],
               "loader": {"minibatch_maxsize": 100},
               "path_for_load_data":
               os.path.join(root.common.test_dataset_root,
                            "arial.ttf"),
               "weights_plotter": {"limit": 16}
               }
