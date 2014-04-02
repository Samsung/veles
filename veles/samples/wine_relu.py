#!/usr/bin/python3.3 -O
"""
Created on August 4, 2013

File for Wine dataset (NN with RELU activation).

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import numpy
import os

from veles.config import root, get_config
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


root.common.update = {"plotters_disabled":
                      get_config(root.common.plotters_disabled, True)}

root.update = {"decision": {"fail_iterations":
                            get_config(root.decision.fail_iterations, 250),
                            "snapshot_prefix":
                            get_config(root.decision.snapshot_prefix,
                                       "wine_relu")},
               "global_alpha": get_config(root.global_alpha, 0.75),
               "global_lambda": get_config(root.global_lambda, 0.0),
               "layers": get_config(root.layers, [10, 3]),
               "loader": {"minibatch_maxsize":
                          get_config(root.loader.minibatch_maxsize, 1000000)},
               "path_for_load_data":
               get_config(root.path_for_load_data,
                          os.path.join(root.common.veles_dir,
                                       "veles/samples/wine/wine.data"))
               }


class Loader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def load_data(self):
        fin = open(root.path_for_load_data, "r")
        aa = []
        max_lbl = 0
        while True:
            s = fin.readline()
            if not len(s):
                break
            aa.append(
                numpy.fromstring(s, sep=",",
                                 dtype=opencl_types.dtypes[root.common.dtype]))
            max_lbl = max(max_lbl, int(aa[-1][0]))
        fin.close()

        self.original_data = numpy.zeros([len(aa), aa[0].shape[0] - 1],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
            [self.original_data.shape[0]],
            dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(max_lbl)])

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
    """Sample workflow for Wine dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self, name="Wine loader",
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllRELU(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.minibatch_max_err_y_sum = self.ev.max_err_y_sum
        self.decision.class_samples = self.loader.class_samples

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDRELU(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.gd[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        self.gd[-1].link_from(self.decision)


def run(load, main):
    load(Workflow, layers=root.layers)
    main()
