#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Approximator config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.decision.fail_iterations = 1000
root.decision.snapshot_prefix = "approximator"
root.decision.store_samples_mse = True
root.global_alpha = 0.01
root.global_lambda = 0.00005
root.layers = [810, 9]
root.loader.minibatch_maxsize = 81
root.path_for_target_data = ["/data/veles/approximator/all_org_appertures.mat"]
root.path_for_train_data = ["/data/veles/approximator/all_dec_appertures.mat"]
