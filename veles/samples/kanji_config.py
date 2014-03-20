#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Kanji config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import six

from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.decision.fail_iterations = 1000
root.decision.snapshot_prefix = "kanji"
root.decision.store_samples_mse = True
root.dir_for_kanji_pickle = "%s/kanji.pickle" % root.common.snapshot_dir
root.global_alpha = 0.001
root.global_lambda = 0.00005
root.layers = [5103, 2889, 24 * 24]
root.loader.minibatch_maxsize = 5103
root.path_for_target_data = "%s/kanji/target/targets.%d.pickle" % (
    root.common.test_dataset_root, 3 if six.PY3 else 2)
root.path_for_train_data = "%s/kanji/train" % (root.common.test_dataset_root)
root.validation_procent = 0.15
root.weights_plotter.limit = 16
