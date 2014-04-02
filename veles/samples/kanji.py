#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for kanji recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import os
import pickle
import six

from veles.config import root, get_config
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader

root.decision.fail_iterations = get_config(root.decision.fail_iterations, 1000)

root.decision.snapshot_prefix = get_config(
    root.decision.snapshot_prefix, "kanji")

root.decision.store_samples_mse = get_config(
    root.decision.store_samples_mse, True)

root.dir_for_kanji_pickle = get_config(
    root.dir_for_kanji_pickle,
    os.path.join(root.common.snapshot_dir, "kanji.pickle"))

root.global_alpha = get_config(root.global_alpha, 0.001)
root.global_lambda = get_config(root.global_lambda, 0.00005)
root.layers = get_config(root.layers, [5103, 2889, 24 * 24])
root.loader.minibatch_maxsize = get_config(root.loader.minibatch_maxsize, 5103)

root.path_for_target_data = get_config(
    root.path_for_target_data,
    os.path.join(root.common.test_dataset_root,
                 ("kanji/target/targets.%d.pickle" % 3 if six.PY3 else 2)))

root.path_for_train_data = get_config(
    root.path_for_train_data,
    os.path.join(root.common.test_dataset_root, "kanji/train"))

root.index_map = get_config(
    root.index_map, os.path.join(root.path_for_train_data,
                                 "index_map.%d.pickle" % 3 if six.PY3 else 2))

root.validation_procent = get_config(root.validation_procent, 0.15)
root.weights_plotter.limit = get_config(root.weights_plotter.limit, 16)


class Loader(loader.Loader):
    """Loads dataset.
    """
    def __init__(self, workflow, **kwargs):
        self.train_path = kwargs["train_path"]
        self.target_path = kwargs["target_path"]
        super(Loader, self).__init__(workflow, **kwargs)
        self.class_target = formats.Vector()

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        state["index_map"] = None
        return state

    def load_data(self):
        """Load the data here.

        Should be filled here:
            class_samples[].
        """
        fin = open(root.index_map, "rb")
        self.index_map = pickle.load(fin)
        fin.close()

        fin = open(os.path.join(self.train_path, self.index_map[0]), "rb")
        self.first_sample = pickle.load(fin)["data"]
        fin.close()

        fin = open(self.target_path, "rb")
        targets = pickle.load(fin)
        fin.close()
        self.class_target.reset()
        sh = [len(targets)]
        sh.extend(targets[0].shape)
        self.class_target.v = numpy.empty(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])
        for i, target in enumerate(targets):
            self.class_target.v[i] = target

        self.label_dtype = opencl_types.itypes[
            opencl_types.get_itype_from_size(len(targets))]

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = len(self.index_map)

        self.original_labels = numpy.empty(len(self.index_map),
                                           dtype=self.label_dtype)
        import re
        lbl_re = re.compile("^(\d+)_\d+/(\d+)\.\d\.pickle$")
        for i, fnme in enumerate(self.index_map):
            res = lbl_re.search(fnme)
            if res is None:
                raise error.ErrBadFormat("Incorrectly formatted filename "
                                         "found: %s" % (fnme))
            lbl = int(res.group(1))
            self.original_labels[i] = lbl
            idx = int(res.group(2))
            if idx != i:
                raise error.ErrBadFormat("Incorrect sample index extracted "
                                         "from filename: %s " % (fnme))

        self.info("Found %d samples. Extracting 15%% for validation..." % (
            len(self.index_map)))
        self.extract_validation_from_train(
            root.validation_procent, rnd.default2)
        self.info("Extracted, resulting datasets are: [%s]" % (
            ", ".join(str(x) for x in self.class_samples)))

    def create_minibatches(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.first_sample.shape)
        self.minibatch_data.v = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        self.minibatch_target.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.class_target.v[0].shape)
        self.minibatch_target.v = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        self.minibatch_labels.reset()
        sh = [self.minibatch_maxsize[0]]
        self.minibatch_labels.v = numpy.zeros(sh, dtype=self.label_dtype)

        self.minibatch_indexes.reset()
        self.minibatch_indexes.v = numpy.zeros(
            len(self.index_map),
            dtype=opencl_types.itypes[opencl_types.get_itype_from_size(
                len(self.index_map))])

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        minibatch_size = self.minibatch_size[0]

        idxs = self.minibatch_indexes.v
        idxs[:minibatch_size] = self.shuffled_indexes[
            self.minibatch_offs[0]:self.minibatch_offs[0] + minibatch_size]

        for i, ii in enumerate(idxs[:minibatch_size]):
            fnme = "%s/%s" % (self.train_path, self.index_map[ii])
            fin = open(fnme, "rb")
            sample = pickle.load(fin)
            data = sample["data"]
            lbl = sample["lbl"]
            fin.close()
            self.minibatch_data.v[i] = data
            self.minibatch_labels.v[i] = lbl
            self.minibatch_target.v[i] = self.class_target[lbl]


class Workflow(workflows.OpenCLWorkflow):
    """Workflow for training network which will be able to recognize
    drawn kanji characters; training done using only TrueType fonts;
    1023 classes to recognize, 3.6 million 32x32 images dataset size.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["name"] = kwargs.get("name", "Kanji")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(
            self, train_path=root.path_for_train_data,
            target_path=root.path_for_target_data)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                     device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(self, device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.class_target = self.loader.class_target
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            store_samples_mse=root.decision.store_samples_mse,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(self, device=device)
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

        self.end_point.link_from(self.gd[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.weights_plotter.limit)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended
        # Max plotter
        self.plt_max = []
        styles = ["", "", "k--"]  # ["r--", "b--", "k--"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.decision)
            self.plt_max[-1].gate_block = ~self.decision.epoch_ended
        # Min plotter
        self.plt_min = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.decision)
            self.plt_min[-1].gate_block = ~self.decision.epoch_ended
        self.plt_min[-1].redraw_plot = True
        # Error plotter
        self.plt_n_err = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_n_err.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt_n_err[-1].input = self.decision.epoch_n_err_pt
            self.plt_n_err[-1].input_field = i
            self.plt_n_err[-1].link_from(self.decision)
            self.plt_n_err[-1].gate_block = ~self.decision.epoch_ended
        self.plt_n_err[0].clear_plot = True
        self.plt_n_err[-1].redraw_plot = True
        # Image plotter
        """
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.decision.vectors_to_sync[self.ev.target] = 1
        self.plt_img = plotters.Image(self, name="output sample")
        self.plt_img.inputs.append(self.decision.sample_input)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.decision.sample_output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.decision.sample_target)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = ~self.decision.epoch_ended
        """
        # Histogram plotter
        self.plt_hist = plotting_units.MSEHistogram(self, name="Histogram")
        self.plt_hist.link_from(self.decision)
        self.plt_hist.mse = self.decision.epoch_samples_mse[2]
        self.plt_hist.gate_block = self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device, weights, bias):
        for g in self.gd:
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
            g.device = device
        for forward in self.forward:
            forward.device = device
        self.ev.device = device
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        super(Workflow, self).initialize()
        if weights is not None:
            for i, forward in enumerate(self.forward):
                forward.weights.map_invalidate()
                forward.weights.v[:] = weights[i][:]
        if bias is not None:
            for i, forward in enumerate(self.forward):
                forward.bias.map_invalidate()
                forward.bias.v[:] = bias[i][:]


def run(load, main):
    weights = None
    bias = None
    w, snapshot = load(Workflow, layers=root.layers)
    if snapshot:
        if type(w) == tuple:
            logging.info("Will load weights")
            weights = w[0]
            bias = w[1]
        else:
            logging.info("Will load workflow")
            logging.info("Weights and bias ranges per layer are:")
            for forward in w.forward:
                logging.info("%f %f %f %f" % (
                    forward.weights.v.min(), forward.weights.v.max(),
                    forward.bias.v.min(), forward.bias.v.max()))
            w.decision.just_snapshotted << True
    main(global_alpha=root.global_alpha, global_lambda=root.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize,
         weights=weights, bias=bias)
