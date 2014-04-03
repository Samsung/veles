#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

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
                            "snapshot_prefix": "cifar"},
               "global_alpha": 0.1,
               "global_lambda": 0.00005,
               "layers_cifar_conv":
                          [{"type": "conv", "n_kernels": 50,
                            "kx": 9, "ky": 9},
                           {"type": "conv", "n_kernels": 100,
                            "kx": 7, "ky": 7},
                           {"type": "conv", "n_kernels": 200,
                            "kx": 5, "ky": 5},
                           {"type": "conv", "n_kernels": 400,
                            "kx": 3, "ky": 3}, 100, 10],
               "loader": {"minibatch_maxsize": 270},
               "path_for_out_data": "/data/veles/cifar/tmpimg/",
               "path_for_train_data":
               os.path.join(root.common.test_dataset_root,
                            "cifar/10"),
               "path_for_valid_data":
               os.path.join(root.common.test_dataset_root,
                            "cifar/10/test_batch"),
               "weights_plotter": {"limit": 25}
               }
