#!/usr/bin/python3.3 -O
"""
Created on October 12, 2013

@author: Seresov Denis <d.seresov@samsung.com>
"""


import numpy

import veles.config as config
from veles.config import getConfig, sconfig
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.rnd as rnd
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


class Loader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def __init__(self,):

        self.loader_input = getConfig(sconfig.loader.input, "")
        # read names file dataset
        self.loader_use_seed = getConfig(sconfig.loader.use_seed, 0)
        self.loader_rnd_seed = getConfig(sconfig.loader.rnd_seed, "")
        self.loader_minibatch_size = getConfig(sconfig.loader.minibatch_size,
                                               100)

        if self.loader_use_seed == 1:
            self.rnd = rnd.Rand()
            self.rnd.seed(numpy.fromfile(self.loader_rnd_seed, numpy.int32,
                                         1024))
        else:
            self.rnd = rnd.Rand()
            self.rnd.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))

        super(Loader, self).__init__(
            minibatch_max_size=self.loader_minibatch_size, rnd=self.rnd)

    def load_data(self):
        """Here we will load MNIST data.
        """
        # global this_dir

        fin = open(self.loader_input, "r")
        aa = []
        max_lbl = 0
        while True:
            s = fin.readline()
            if not len(s):
                break
            aa.append(numpy.fromstring(s, sep=",",
                                 dtype=opencl_types.dtypes[config.dtype]))
            max_lbl = max(max_lbl, int(aa[-1][0]))
        fin.close()

        self.original_data = numpy.zeros(
                                         [len(aa), aa[0].shape[0] - 1],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
                [self.original_data.shape[0]],
                dtype=opencl_types.itypes[opencl_types.\
                           get_itype_from_size(max_lbl)])

        for i, a in enumerate(aa):
            self.original_data[i] = a[1:]
            self.original_labels[i] = int(a[0]) - 1
            # formats.normalize(self.original_data[i])

        IMul, IAdd = formats.normalize_pointwise(self.original_data)
        self.original_data[:] *= IMul
        self.original_data[:] += IAdd

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = self.original_data.shape[0]

        self.nextclass_offs[0] = 0
        self.nextclass_offs[1] = 0
        self.nextclass_offs[2] = self.original_data.shape[0]

        self.total_samples[0] = self.original_data.shape[0]


class Workflow(workflows.OpenCLWorkflow):
    """Sample workflow for MNIST dataset.
    """
    def wait_finish(self):
        # if use plotters
        # plotters.Graphics().wait_finish()
        print('plotters.Graphics().wait_finish()   XZ!')

    def __init__(self):
        """
        Workflow for NN use config.snapshot_prefix ,snapshot, ..
         and t.e. parametrs for experiments
        Workflow for NN use config.wf_nn
        Workflow for NN use config.wf_nn_train
        """
        self.snapshot_prefix = getConfig(sconfig.snapshot_prefix, 'test_pr')
        self.snapshot = getConfig(sconfig.snapshot, 'test')

        self.nn_layers = getConfig(sconfig.wf_nn.layers, 0)
        self.weights_amplitude = getConfig(
                        sconfig.wf_nn.weights_amplitude, None)
        self.weights_amplitude_type = getConfig(
                        sconfig.wf_nn.weights_amplitude_type, 0)
        """
         choice of method of generation of weights
          0 - self.weights_amplitude =
                          min (self.get_weights_amplitude (), 0.05)
          1 - fix Range of
                    [-self.weights_amplitude; self.weights_amplitude]
          2 other methods (add)
         default use wf_nn.weights_amplitude = None and use method=0
        """

        self.wf_nn_train_fail_iterations = getConfig(
                        sconfig.wf_nn_train.fail_iterations, 25)
        self.train_global_alpha = getConfig(
                        sconfig.wf_nn_train.global_alpha, 0.1)
        print(self.train_global_alpha)
        self.train_global_lambda = getConfig(
                        sconfig.wf_nn_train.global_lambda, 0.0)
        self.train_momentum = getConfig(
                        sconfig.wf_nn_train.momentum, 0.9)

        self.wf_nn_use_seed = getConfig(sconfig.wf_nn.use_seed, 0)
        self.wf_nn_rnd_seed = getConfig(sconfig.wf_nn.rnd_seed, "")

        self.compute_confusion_matrix = getConfig(
                        sconfig.wf_nn_train.compute_confusion_matrix, 1)
        """ (not work) """

        # initialization device ... may be refactor
        self.device = sconfig.device

        # validation errors parameters
        if self.nn_layers == 0:
            print('error NN layers')

        # initialization random generator for nn
        # need  be connected to random weigths
        if self.wf_nn_use_seed == 1:
            self.rnd = rnd.Rand()
            self.rnd.seed(numpy.fromfile(
                        self.wf_nn_rnd_seed, numpy.int32, 1024))
        else:
            self.rnd = rnd.Rand()
            self.rnd.seed(numpy.fromfile(
                        "/dev/urandom", numpy.int32, 1024))

        # self.device = args.device
        # self.args.snapshot_prefix =args.snapshot_prefix; #'wine'

        super(Workflow, self).__init__(device=self.device)

        self.rpt.link_from(self.start_point)

        # Loader use config.loader
        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
        for i in range(0, len(self.nn_layers)):
            if i < len(self.nn_layers) - 1:
                aa = all2all.All2AllTanh([self.nn_layers[i]],
                                    device=self.device,
                                    weights_amplitude=self.weights_amplitude,
                                    rand=self.rnd)
            else:
                aa = all2all.All2AllSoftmax([self.nn_layers[i]],
                                    device=self.device,
                                    weights_amplitude=self.weights_amplitude,
                                    rand=self.rnd)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self,
                        device=self.device,
                        compute_confusion_matrix=self.compute_confusion_matrix)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(self,
                            fail_iterations=self.wf_nn_train_fail_iterations,
                            snapshot_prefix=self.snapshot_prefix)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.minibatch_max_err_y_sum = self.ev.max_err_y_sum
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=self.device)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=self.device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        self.gd[-1].link_from(self.decision)

    def initialize(self):
        for g in self.gd:
            g.global_alpha = self.train_global_alpha
            g.global_lambda = self.train_global_lambda
        super(Workflow, self).initialize(device=self.device)
        return super(Workflow, self).initialize()
