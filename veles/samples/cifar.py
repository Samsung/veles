#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

Cifar all2all.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import os
import pickle

from veles.config import root, get_config
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader

root.update = {"decision": {"fail_iterations":
                            get_config(root.decision.fail_iterations, 100),
                            "snapshot_prefix":
                            get_config(root.decision.snapshot_prefix,
                                       "cifar")},
               "global_alpha": get_config(root.global_alpha, 0.1),
               "global_lambda": get_config(root.global_lambda, 0.00005),
               "layers": get_config(root.layers, [100, 10]),
               "loader": {"minibatch_maxsize":
                          get_config(root.loader.minibatch_maxsize, 180)},
               "path_for_out_data": get_config(root.path_for_out_data,
                                               "/data/veles/cifar/tmpimg/"),
               "path_for_train_data":
               get_config(root.path_for_train_data,
                          os.path.join(root.common.test_dataset_root,
                                       "cifar/10")),
               "path_for_valid_data":
               get_config(root.path_for_valid_data,
                          os.path.join(root.common.test_dataset_root,
                                       "cifar/10/test_batch")),
               "weights_plotter": {"limit":
                                   get_config(root.weights_plotter.limit, 25)}
               }


class Loader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def load_data(self):
        """Here we will load data.
        """
        n_classes = 10
        self.original_data = numpy.zeros([60000, 32, 32, 3],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
            60000, dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(n_classes)])

        # Load Validation
        fin = open(root.path_for_valid_data, "rb")
        u = pickle._Unpickler(fin)
        u.encoding = 'latin1'
        vle = u.load()
        fin.close()
        self.original_data[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            fin = open(os.path.join(root.path_for_train_data,
                                    ("data_batch_%d" % i)), "rb")
            u = pickle._Unpickler(fin)
            u.encoding = 'latin1'
            vle = u.load()
            fin.close()
            self.original_data[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels[i * 10000: (i + 1) * 10000] = vle["labels"][:]

        self.class_samples[0] = 0
        self.nextclass_offs[0] = 0
        self.class_samples[1] = 10000
        self.nextclass_offs[1] = 10000
        self.class_samples[2] = 50000
        self.nextclass_offs[2] = 60000

        self.total_samples[0] = self.original_data.shape[0]

        for sample in self.original_data:
            formats.normalize(sample)


class Workflow(workflows.OpenCLWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self,
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
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

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self, out_dirs=[
            os.path.join(root.path_for_out_data, "test"),
            os.path.join(root.path_for_out_data, "validation"),
            os.path.join(root.path_for_out_data, "train")])
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.max_idx = self.forward[-1].max_idx
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.image_saver)
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
        self.decision.class_samples = self.loader.class_samples

        self.image_saver.gate_skip = ~self.decision.just_snapshotted
        self.image_saver.snapshot_time = self.decision.snapshot_time

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
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
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Matrix plotter
        # """
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_w.input = self.gd[0].weights
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.decision)
        self.plt_w.gate_block = ~self.decision.epoch_ended
        # """
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt[-1])
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device):
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    load(Workflow, layers=root.layers)
    main(global_alpha=root.global_alpha, global_lambda=root.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
