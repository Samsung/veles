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
                            "snapshot_prefix": "mnist_conv"},
               "global_alpha": 0.005,
               "global_lambda": 0.00005,
               "layers_mnist_conv":
               [{"type": "conv", "n_kernels": 25, "kx": 9, "ky": 9}, 100, 10],
               "loader": {"minibatch_maxsize": 540},
               "path_for_load_data":
               os.path.join(root.common.test_dataset_root,
                            "arial.ttf"),
               "weights_plotter": {"limit": 64}
               }
