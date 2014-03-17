#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for kanji recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import pickle
import os
import sys

import veles.config as config
import veles.error as error
import veles.formats as formats
import veles.launcher as launcher
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


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
        fin = open("%s/index_map.pickle" % (self.train_path), "rb")
        self.index_map = pickle.load(fin)
        fin.close()

        fin = open("%s/%s" % (self.train_path, self.index_map[0]), "rb")
        self.first_sample = pickle.load(fin)["data"]
        fin.close()

        fin = open(self.target_path, "rb")
        targets = pickle.load(fin)
        fin.close()
        self.class_target.reset()
        sh = [len(targets)]
        sh.extend(targets[0].shape)
        self.class_target.v = numpy.empty(sh,
            dtype=opencl_types.dtypes[config.c_dtype])
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
        lbl_re = re.compile("^(\d+)_\d+/(\d+)\.pickle$")
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
        self.extract_validation_from_train(0.15, rnd.default2)
        self.info("Extracted, resulting datasets are: [%s]" % (
            ", ".join(str(x) for x in self.class_samples)))

    def create_minibatches(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.first_sample.shape)
        self.minibatch_data.v = numpy.zeros(sh,
                dtype=opencl_types.dtypes[config.c_dtype])

        self.minibatch_target.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.class_target.v[0].shape)
        self.minibatch_target.v = numpy.zeros(sh,
            dtype=opencl_types.dtypes[config.c_dtype])

        self.minibatch_labels.reset()
        sh = [self.minibatch_maxsize[0]]
        self.minibatch_labels.v = numpy.zeros(sh, dtype=self.label_dtype)

        self.minibatch_indexes.reset()
        self.minibatch_indexes.v = numpy.zeros(len(self.index_map),
            dtype=opencl_types.itypes[opencl_types.get_itype_from_size(
                                                        len(self.index_map))])

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        minibatch_size = self.minibatch_size[0]

        idxs = self.minibatch_indexes.v
        idxs[:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:
            self.minibatch_offs[0] + minibatch_size]

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


import veles.workflows as workflows


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

        self.loader = Loader(self,
            train_path="%s/kanji/train" % (config.test_dataset_root),
            target_path="%s/kanji/target/targets.pickle" % (
                                            config.test_dataset_root))
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
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
        self.decision = decision.Decision(self, store_samples_mse=True,
                                          snapshot_prefix="kanji")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self
        self.decision.should_unlock_pipeline = False

        # Add gradient descent units
        self.gd.clear()
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
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotting_units.SimplePlotter(self, name="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(self,
                                         name="First Layer Weights",
                                         limit=16)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = self.decision.epoch_ended
        self.plt_mx.gate_block_not = [1]
        # Max plotter
        self.plt_max = []
        styles = ["", "", "k--"]  # ["r--", "b--", "k--"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotting_units.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.decision)
            self.plt_max[-1].gate_block = self.decision.epoch_ended
            self.plt_max[-1].gate_block_not = [1]
        # Min plotter
        self.plt_min = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotting_units.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.decision)
            self.plt_min[-1].gate_block = self.decision.epoch_ended
            self.plt_min[-1].gate_block_not = [1]
        self.plt_min[-1].redraw_plot = True
        # Error plotter
        self.plt_n_err = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_n_err.append(plotting_units.SimplePlotter(self,
                                name="num errors", plot_style=styles[i]))
            self.plt_n_err[-1].input = self.decision.epoch_n_err_pt
            self.plt_n_err[-1].input_field = i
            self.plt_n_err[-1].link_from(self.decision)
            self.plt_n_err[-1].gate_block = self.decision.epoch_ended
            self.plt_n_err[-1].gate_block_not = [1]
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
        self.plt_img.gate_block = self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]
        """
        # Histogram plotter
        self.plt_hist = plotting_units.MSEHistogram(self, name="Histogram")
        self.plt_hist.link_from(self.decision)
        self.plt_hist.mse = self.decision.epoch_samples_mse[2]
        self.plt_hist.gate_block = self.decision.epoch_ended
        self.plt_hist.gate_block_not = [1]
        self.plt_hist.should_unlock_pipeline = True

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device, weights, bias):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
            gd.device = device
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


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    rnd.default.seed(numpy.fromfile("%s/seed" % (os.path.dirname(__file__)),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 524288))
    l = launcher.Launcher()
    device = None if l.is_master else opencl.Device()
    fnme = "%s/kanji.pickle" % (config.snapshot_dir)
    fin = None
    try:
        fin = open(fnme, "rb")
    except IOError:
        pass
    weights = None
    bias = None
    if fin is not None:
        w = pickle.load(fin)
        fin.close()
        if type(w) == tuple:
            logging.info("Will load weights")
            weights = w[0]
            bias = w[1]
            fin = None
        else:
            logging.info("Will load workflow")
            logging.info("Weights and bias ranges per layer are:")
            for forward in w.forward:
                logging.info("%f %f %f %f" % (
                    forward.weights.v.min(), forward.weights.v.max(),
                    forward.bias.v.min(), forward.bias.v.max()))
            w.decision.just_snapshotted[0] = 1
    if fin is None:
        w = Workflow(l, layers=[5103, 2889, 24 * 24], device=device)
    w.initialize(global_alpha=0.001, global_lambda=0.00005,
                 minibatch_maxsize=5103, device=device,
                 weights=weights, bias=bias)
    l.run()

    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
