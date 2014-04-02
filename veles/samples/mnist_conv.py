#!/usr/bin/python3.3 -O
"""
Created on November 25, 2013

MNIST with Convolutional layer.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


from veles.config import root, get_config
import veles.error as error
import veles.plotting_units as plotting_units
from veles.samples import mnist as mnist
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling


root.update = {"decision": {"fail_iterations":
                            get_config(root.decision.fail_iterations, 100),
                            "snapshot_prefix":
                            get_config(root.decision.snapshot_prefix,
                                       "mnist_conv")},
               "global_alpha": get_config(root.global_alpha, 0.005),
               "global_lambda": get_config(root.global_lambda, 0.00005),
               "layers_mnist_conv":
               get_config(root.layers_mnist_conv,
                          [{"type": "conv", "n_kernels": 25, "kx": 9, "ky": 9},
                           100, 10]),
               "loader": {"minibatch_maxsize":
                          get_config(root.loader.minibatch_maxsize, 540)},
               "weights_plotter": {"limit":
                                   get_config(root.weights_plotter.limit, 64)}}


class Workflow(workflows.OpenCLWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    A deep learning method (advanced convolutional neural network) is used.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = "Convolutional MNIST"
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = mnist.Loader(
            self, minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            layer = layers[i]
            if type(layer) == int:
                if i == len(layers) - 1:
                    aa = all2all.All2AllSoftmax(self, output_shape=[layer],
                                                device=device)
                else:
                    aa = all2all.All2AllTanh(self, output_shape=[layer],
                                             device=device)
            elif type(layer) == dict:
                if layer["type"] == "conv":
                    aa = conv.ConvTanh(
                        self, n_kernels=layer["n_kernels"],
                        kx=layer["kx"], ky=layer["ky"], device=device)
                elif layer["type"] == "max_pooling":
                    aa = pooling.MaxPooling(
                        self, kx=layer["kx"], ky=layer["ky"], device=device)
                elif layer["type"] == "avg_pooling":
                    aa = pooling.AvgPooling(
                        self, kx=layer["kx"], ky=layer["ky"], device=device)
                else:
                    raise error.ErrBadFormat(
                        "Unsupported layer type %s" % (layer["type"]))
            else:
                raise error.ErrBadFormat(
                    "layers element type should be int "
                    "for all-to-all or dictionary for "
                    "convolutional or pooling")
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
        self.gd.extend(list(None for i in range(0, len(self.forward))))
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
            if isinstance(self.forward[i], conv.Conv):
                obj = gd_conv.GDTanh(
                    self, n_kernels=self.forward[i].n_kernels,
                    kx=self.forward[i].kx, ky=self.forward[i].ky,
                    device=device)
            elif isinstance(self.forward[i], pooling.MaxPooling):
                obj = gd_pooling.GDMaxPooling(
                    self, kx=self.forward[i].kx, ky=self.forward[i].ky,
                    device=device)
                obj.h_offs = self.forward[i].input_offs
            elif isinstance(self.forward[i], pooling.AvgPooling):
                obj = gd_pooling.GDAvgPooling(
                    self, kx=self.forward[i].kx, ky=self.forward[i].ky,
                    device=device)
            else:
                obj = gd.GDTanh(self, device=device)
            self.gd[i] = obj
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

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        # err_y plotter
        self.plt_err_y = []
        for i in range(1, 3):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].input = self.decision.max_err_y_sums
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision)
            self.plt_err_y[-1].gate_block = ~self.decision.epoch_ended
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = (
            [self.forward[0].kx, self.forward[0].ky]
            if isinstance(self.forward[0], conv.Conv)
            else self.forward[0].input)
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device):
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        for f in self.forward:
            f.device = device
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        return super(Workflow, self).initialize(device=device)


def run(load, main):
    load(layers=root.layers_mnist_conv)
    """
    W = []
    b = []
    for f in w.forward:
        W.append(f.weights.v)
        b.append(f.bias.v)
    fout = open("/tmp/Wb.pickle", "wb")
    pickle.dump((W, b), fout)
    fout.close()
    sys.exit(0)
    """
    # w = Workflow(None, layers=[
    #                     {"type": "conv", "n_kernels": 25, "kx": 9, "ky": 9},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 0.98%
    #                     100, 10], device=device)
    # w = Workflow(None, layers=[
    #                     {"type": "conv", "n_kernels": 50, "kx": 9, "ky": 9},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 10
    #                     {"type": "conv", "n_kernels": 200, "kx": 3, "ky": 3},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 4
    #                     100, 10], device=device)
    main(global_alpha=root.global_alpha,
         global_lambda=root.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
