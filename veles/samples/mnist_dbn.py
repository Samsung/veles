#!/usr/bin/python3.3 -O
"""
Created on Dec 23, 2013

MNIST via DBN.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
from freetype import *
import logging
import numpy
import pickle
import re
import os
import sys

import veles.config as config
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.samples.mnist as mnist
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.rbm as rbm

no_random = True
autoencoder = False


def do_plot(fontPath, text, size, angle, sx, sy,
            randomizePosition, SX, SY):
    face = Face(bytes(fontPath, 'UTF-8'))
    # face.set_char_size(48 * 64)
    face.set_pixel_sizes(0, size)

    c = text[0]

    angle = (angle / 180.0) * numpy.pi

    mx_r = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                        [numpy.sin(angle), numpy.cos(angle)]],
                       dtype=numpy.double)
    mx_s = numpy.array([[sx, 0.0],
                        [0.0, sy]], dtype=numpy.double)

    mx = numpy.dot(mx_s, mx_r)

    matrix = FT_Matrix((int)(mx[0, 0] * 0x10000),
                       (int)(mx[0, 1] * 0x10000),
                       (int)(mx[1, 0] * 0x10000),
                       (int)(mx[1, 1] * 0x10000))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0, 0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))

    j = 0
    while True:
        slot = face.glyph
        if not face.get_char_index(c):
            return None
        face.load_char(c, flags)
        bitmap = slot.bitmap
        width = bitmap.width
        height = bitmap.rows
        if width > SX or height > SY:
            j = j + 1
            face.set_pixel_sizes(0, size - j)
            # logging.info("Set pixel size for font %s to %d" % (
            #    fontPath, size - j))
            continue
        break

    if randomizePosition:
        x = int(numpy.floor(numpy.random.rand() * (SX - width)))
        y = int(numpy.floor(numpy.random.rand() * (SY - height)))
    else:
        x = int(numpy.floor((SX - width) * 0.5))
        y = int(numpy.floor((SY - height) * 0.5))

    img = numpy.zeros([SY, SX], dtype=numpy.uint8)
    img[y:y + height, x: x + width] = numpy.array(bitmap.buffer,
        dtype=numpy.uint8).reshape(height, width)
    if img.max() == img.min():
        logging.info("Font %s returned empty glyph" % (fontPath))
        return None
    return img


class Loader(mnist.Loader):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Loader, self).load_data()

        self.class_target.reset()
        # self.class_target.v = numpy.zeros([10, 10],
        #    dtype=opencl_types.dtypes[config.dtype])
        self.class_target.v = numpy.zeros([10, 784],
            dtype=opencl_types.dtypes[config.dtype])

        for i in range(10):
            # self.class_target.v[i, :] = -1
            # self.class_target.v[i, i] = 1
            img = do_plot("%s/arial.ttf" % (config.test_dataset_root),
                          "%d" % (i,), 28, 0.0, 1.0, 1.0, False, 28, 28)
            self.class_target.v[i] = img.ravel().astype(
                                opencl_types.dtypes[config.dtype])
            formats.normalize(self.class_target.v[i])

        self.original_target = numpy.zeros([self.original_labels.shape[0],
                                            self.class_target.v.shape[1]],
            dtype=self.original_data.dtype)
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target.v[label]

        # At the beginning, initialize "values to be found" with zeros.
        # NN should be trained in the same way as it will be tested.
        # v = self.original_data
        # v = v.reshape(v.shape[0], v.size // v.shape[0])
        # self.original_data = numpy.zeros([v.shape[0],
        #    self.class_target.v.shape[1] + v.shape[1]], dtype=v.dtype)
        # self.original_data[:, self.class_target.v.shape[1]:] = v[:, :]


class Workflow(workflows.OpenCLWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        recursion_depth = kwargs.get("recursion_depth")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["recursion_depth"] = recursion_depth
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)
        self.rpt.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.rpt)

        self.recursion_depth = recursion_depth
        for i in range(recursion_depth):
            if no_random:
                aa = all2all.All2AllTanh(self, output_shape=[layers[-1]],
                                         device=device)
            else:
                aa = rbm.RBMTanh(self, output_shape=[layers[-1]],
                                 device=device)
            self.forward.append(aa)
            if i:
                self.forward[-1].link_from(self.forward[-2])
                self.forward[-1].input = self.forward[-2].output
                self.forward[-1].weights = self.forward[-3].weights
                self.forward[-1].bias = self.forward[-3].bias
            else:
                self.forward[-1].link_from(self.loader)
                self.forward[-1].input = self.loader.minibatch_data

            aa = all2all.All2AllTanh(self, output_shape=self.forward[-1].input,
                device=device, weights_transposed=autoencoder)
            self.forward.append(aa)
            self.forward[-1].link_from(self.forward[-2])
            self.forward[-1].input = self.forward[-2].output
            if autoencoder:
                self.forward[-1].weights = self.forward[-2].weights
            if i:
                self.forward[-1].weights = self.forward[-3].weights
                self.forward[-1].bias = self.forward[-3].bias

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(self, device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.max_samples_per_epoch = self.loader.total_samples
        self.ev.class_target = self.loader.class_target
        self.ev.labels = self.loader.minibatch_labels

        # Add decision unit
        self.decision = decision.Decision(self, snapshot_prefix="mnist_dbn",
                                          store_samples_mse=True)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(len(self.forward)))
        self.gd[-1] = gd.GDTanh(self, device=device,
                                weights_transposed=autoencoder)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device,
                weights_transposed=(autoencoder and (i & 1)))
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
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(self,
                                         name="First Layer Weights",
                                         limit=100)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = self.forward[0].input
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = self.decision.epoch_ended
        self.plt_mx.gate_block_not = [1]
        # Image plotter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.decision.vectors_to_sync[self.ev.target] = 1
        self.plt_img = plotting_units.Image(self, name="output sample")
        self.plt_img.inputs.append(self.decision.sample_input)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.decision.sample_output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.decision.sample_target)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]
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
        self.decision.snapshot_prefix = args.snapshot_prefix
        for g in self.gd:
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
    parser.add_argument("-snapshot_prefix", type=str, required=True,
        help="Snapshot prefix (Ex.: mnist_dbn_2000)")
    parser.add_argument("-layers", type=str, required=True,
        help="NN layer sizes, separated by any separator (Ex.: 2000)")
    parser.add_argument("-minibatch_size", type=int,
        help="Minibatch size (default 108)", default=108)
    parser.add_argument("-recursion_depth", type=int,
        help="Depth of the RBM's recursive pass (default 1)", default=1)
    parser.add_argument("-global_alpha", type=float,
        help="Global Alpha (default 0.001)", default=0.001)
    parser.add_argument("-global_lambda", type=float,
        help="Global Lambda (default 0.00005)", default=0.00005)
    args = parser.parse_args()

    s_layers = re.split("\D+", args.layers)
    layers = []
    for s in s_layers:
        layers.append(int(s))
    logging.info("Will train NN with layers: %s" % (" ".join(
                                        str(x) for x in layers)))

    rnd.default.seed(numpy.fromfile("%s/seed" % (os.path.dirname(__file__)),
                                    numpy.int32, 1024))
    rnd.default2.seed(numpy.fromfile("%s/seed2" % (os.path.dirname(__file__)),
                                    numpy.int32, 1024))
    device = opencl.Device()
    try:
        fin = open(args.snapshot, "rb")
        w = pickle.load(fin)
        fin.close()
    except IOError:
        w = Workflow(None, layers=layers, recursion_depth=args.recursion_depth,
                     device=device)
    w.initialize(device=device, args=args)
    w.run()

    logging.debug("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
