#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

MNIST with RBM pretraining.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
import logging
import numpy
import os
import pickle
import sys

import veles.opencl as opencl
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.samples.mnist as mnist
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.rbm as rbm


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

        self.loader = mnist.Loader(self)
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
        for i in range(len(layers)):
            if i < len(layers) - 1:
                aa = rbm.RBMTanh(self, output_shape=[layers[i]], device=device)
            else:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                    device=device, weights_transposed=True)
                aa.weights = self.forward[-1].weights
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
        self.ev.target = self.forward[-2].input
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(self, fail_iterations=25,
                                          snapshot_prefix="mnist_rbm",
                                          store_samples_mse=True)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(self, device=device, weights_transposed=True)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        last_gd = self.gd[-1]
        for i in range(len(self.forward) - 2, len(self.forward) - 3, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
            last_gd = self.gd[i]
        self.rpt.link_from(last_gd)

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.SimplePlotter(self, name="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended if not i
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[-1].weights] = 1
        self.plt_mx = []
        self.plt_mx.append(
            plotting_units.Weights2D(self, name="Last Layer Weights",
                                     limit=64))
        self.plt_mx[-1].input = self.gd[-1].weights
        self.plt_mx[-1].input_field = "v"
        # self.plt_mx[-1].get_shape_from = self.forward[0].input
        self.plt_mx[-1].link_from(self.decision)
        self.plt_mx[-1].gate_block = self.decision.epoch_ended
        self.plt_mx[-1].gate_block_not = [1]
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True
        # Image plotter
        self.decision.vectors_to_sync[self.forward[-2].input] = 1
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.plt_img = plotting_units.Image(self, name="sample")
        self.plt_img.inputs.append(self.forward[-2].input)
        self.plt_img.input_fields.append("v")
        self.plt_img.inputs.append(self.forward[-1].output)
        self.plt_img.input_fields.append("v")
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_skip = self.decision.epoch_ended
        self.plt_img.gate_skip_not = [1]
        self.gd[-1].link_from(self.plt_img)
        # Histogram plotter
        self.plt_hist = [None,
            plotting_units.MSEHistogram(self, name="Histogram Validation"),
            plotting_units.MSEHistogram(self, name="Histogram Train")]
        self.plt_hist[1].link_from(self.decision)
        self.plt_hist[1].mse = self.decision.epoch_samples_mse[1]
        self.plt_hist[1].gate_block = self.decision.epoch_ended
        self.plt_hist[1].gate_block_not = [1]
        self.plt_hist[2].link_from(self.plt_hist[1])
        self.plt_hist[2].mse = self.decision.epoch_samples_mse[2]

    def initialize(self, device, args):
        for g in self.gd:
            if g is None:
                continue
            g.global_alpha = args.global_alpha
            g.global_lambda = args.global_lambda
        return super(Workflow, self).initialize(device=device)


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, default="",
        help="Snapshot with trained network (default empty)")
    parser.add_argument("-global_alpha", type=float,
        help="Global Alpha (default 0.0001)", default=0.0001)
    parser.add_argument("-global_lambda", type=float,
        help="Global Lambda (default 0.00005)", default=0.00005)
    args = parser.parse_args()

    rnd.default.seed(numpy.fromfile("%s/seed" % (os.path.dirname(__file__)),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    W = []
    b = []
    device = opencl.Device()
    try:
        fin = open(args.snapshot, "rb")
        W, b = pickle.load(fin)
        fin.close()
    except IOError:
        pass
    layers = []
    for i in range(len(W) - 1):
        layers.append(len(b[i]))
    layers.append(529)
    if len(layers) == 1:
        layers.append(784)
    else:
        layers.append(layers[-2])
    logging.info("Will train with layers: %s" % (str(layers)))
    w = Workflow(None, layers=layers, device=device)
    w.initialize(device=device, args=args)
    if len(W):
        logging.info("Will set weights to pretrained ones")
        for i in range(len(W) - 1):
            w.forward[i].weights.map_invalidate()
            w.forward[i].weights.v[:] = W[i][:]
            w.forward[i].bias.map_invalidate()
            w.forward[i].bias.v[:] = b[i][:]
        logging.info("Done")
    w.run()

    logging.debug("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
