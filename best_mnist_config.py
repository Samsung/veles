#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root

mnist_dir = mnist_dir = "veles/znicz/samples/MNIST"

# optional parameters
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.update = {"all2all": {"weights_stddev": 0.05},
               "decision": {"fail_iterations": 300,
                            "snapshot_prefix": "mnist",
			    "max_epochs": 70},
               "loader": {"minibatch_size": 88},#88
               "mnist": {"learning_rate": 0.028557478339518444, # 0.028557478339518444
                         "weights_decay": 0.00012315096341168246, #0.00012315096341168246,
                         "factor_ortho": 0.001,  # 1.52%
                         "layers": [364, 10], # 364
                         "data_paths": {"test_images": test_image_dir, #1.81% err
                                        "test_label": test_label_dir,
                                        "train_images": train_image_dir,
                                        "train_label": train_label_dir}}}
