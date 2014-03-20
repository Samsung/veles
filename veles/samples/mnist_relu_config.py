#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os
from veles.config import root, Config

mnist_dir = "/data/veles/Veles2/veles/samples/MNIST"

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters
root.update = {"all2all": {"weights_magnitude": 0.05},
               "decision": {"fail_iterations": 150,
                            "snapshot_prefix": "mnist_relu"},
               "global_alpha": 0.01,
               "global_lambda": 0.0,
               "layers_mnist_relu": [100, 10],
               "loader": {"minibatch_maxsize": 60},
               "path_for_load_data_test_images":
               os.path.join(mnist_dir, "t10k-images.idx3-ubyte"),
               "path_for_load_data_test_label":
               os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"),
               "path_for_load_data_train_images":
               os.path.join(mnist_dir, "train-images.idx3-ubyte"),
               "path_for_load_data_train_label":
               os.path.join(mnist_dir, "train-labels.idx1-ubyte")}
