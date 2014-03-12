#!/usr/bin/python3.3 -O
"""
Created on October 12, 2013

@author: Seresov Denis <d.seresov@samsung.com>
"""


import os
import veles.config as config
import veles.opencl as opencl

from veles.config import Config, sconfig
sconfig.device = opencl.Device()
sconfig.name_data = 'wine'
this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."

sconfig.wf = 'wine_2_workflow'
sconfig.snapshot_prefix = "%s_2_" % (sconfig.name_data)
sconfig.use_snapshot = 0
sconfig.snapshot = "%s/wine_2_.pickle" % (config.snapshot_dir)

sconfig.loader = Config()
sconfig.wf_nn = Config()
sconfig.wf_nn_train = Config()

sconfig.loader.input = "%s/wine/wine.data" % (this_dir)
sconfig.loader.use_seed = 1
"""
0 - not seed
1 - file seed
2 - vector [] or int (not work)
"""
sconfig.loader.rnd_seed = "%s/seed" % (this_dir)
"""sconfig.loader.minibatch_use = 1"""
sconfig.loader.minibatch_size = 6
""" fold would be the best size of
the matrix multiplication in this opencl device"""

sconfig.wf_nn.layers = [8, 3]
sconfig.wf_nn.weights_amplitude = 0.1
sconfig.wf_nn.weights_amplitude_type = 0
sconfig.wf_nn.use_seed = 1
"""
0 - not seed
1 - file seed
2 - vector [] or int (not work)
"""
sconfig.wf_nn.rnd_seed = "%s/seed" % (this_dir)
#sconfig.wf_nn_train.global_alpha = 0.2
sconfig.wf_nn_train.momentum = 0.8
""" sconfig.wf_nn_train.global_alpha +sconfig.wf_nn_train.momentum =1  """
sconfig.wf_nn_train.global_lambda = 0.02
sconfig.wf_nn_train.fail_iterations = 10

sconfig.compute_confusion_matrix = 1
""" 0 - not use 1 - use () ||
   -(use Evaluator and self.decision.minibatch_confusion_matrix)
   - enable/disable plotter confusion_matrix

  (not work for plotter)
   - enable/disable others plotter
     (not work for plotter)
"""
#self.use_regulate_weigths_=0;
#self.use_delete_small_weights=0;
#self.use_normalize_weigths=0;
