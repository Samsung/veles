#!/usr/bin/python3.3 -O

"""
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root


# optional parameters
root.common.snapshot_dir = "1992"
root.update = {"learning_rate_adjust": {"do": True}, # True False
               "decision": {"max_epochs": 1000},
               "snapshotter": {"prefix": "mnist_caffe"},
               "loader": {"minibatch_size": 6},
               "weights_plotter": {"limit": 64},
               "mnist": {#"learning_rate": 0.01, "gradient_moment": 0.9, # 0-1
                         #"weights_decay": 0.0005,
                         "layers":
                         [{"type": "conv", # conv conv_relu conv_str
                           "n_kernels": 64,
                           "kx": 5, "ky": 5,
                           "sliding": (1, 1),
                           "learning_rate": 0.466000,
                           "learning_rate_bias": 0.358000,
                           "gradient_moment": 0.36508255921752014,
                           "gradient_moment_bias": 0.385000,
                           "weights_filling": "uniform", # "gaussian"
                           "weights_stddev": 0.0944569801138958,
                           "bias_filling": "constant", # "uniform", "gaussian"
                           "bias_stddev": 0.048000,
                           "weights_decay": 0.38780014161121407,
                           "weights_decay_bias": 0.1980997902551238
                           },
                          {"type": "max_pooling",# abs_pooling
                           "kx": 2, "ky": 2,
                           "sliding": (2, 2)},


                          {"type": "conv",
                           "n_kernels": 87,
                           "kx": 5, "ky": 5,
                           "sliding": (1, 1),
                           "learning_rate": 0.027000,
                           "learning_rate_bias": 0.381000,
                           "gradient_moment": 0.115000,
                           "gradient_moment_bias": 0.741000,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.067000,
                           "bias_filling": "constant",
                           "bias_stddev": 0.444000,
                           "weights_decay": 0.286000,
                           "weights_decay_bias": 0.039000
                           },
                          {"type": "max_pooling",
                           "kx": 2, "ky": 2, "sliding": (2, 2)},

                          {"type": "all2all_relu",
                           "output_shape": 791, # 10 - 1000
                           "learning_rate": 0.039000,
                           "learning_rate_bias": 0.196000,
                           "gradient_moment": 0.810000,
                           "gradient_moment_bias": 0.619000,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.039000,
                           "bias_filling": "constant",
                           "bias_stddev": 1.000000,
                           "weights_decay": 0.110000,
                           "weights_decay_bias": 0.11487830567238211},
                          {"type": "softmax",
                           "output_shape": 10,
                           "learning_rate": 0.342000,
                           "learning_rate_bias": 0.488000,
                           "gradient_moment": 0.133000,
                           "gradient_moment_bias": 0.8422143625658985,
                           "weights_filling": "uniform",
                           "weights_stddev": 0.024000,
                           "bias_filling": "constant",
                           "bias_stddev": 0.255000,
                           "weights_decay": 0.356000,
                           "weights_decay_bias": 0.476000}]}}
