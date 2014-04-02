#!/usr/bin/python3.3 -O
"""
Created on August 12, 2013

MNIST with target encoded as ideal image (784 points), MSE.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


from freetype import (Face, FT_Matrix,
                      FT_LOAD_RENDER, FT_Vector, FT_Set_Transform, byref)
import logging
import numpy
import os

from veles.config import root, get_config
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.samples.mnist as mnist
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver


root.update = {"decision": {"fail_iterations":
                            get_config(root.decision.fail_iterations, 100),
                            "snapshot_prefix":
                            get_config(root.decision.snapshot_prefix,
                                       "mnist_784")},
               "global_alpha": get_config(root.global_alpha, 0.001),
               "global_lambda": get_config(root.global_lambda, 0.00005),
               "layers_mnist784": get_config(root.layers_mnist784, [784, 784]),
               "loader": {"minibatch_maxsize":
                          get_config(root.loader.minibatch_maxsize, 100)},
               "path_for_load_data":
               get_config(root.path_for_load_data,
                          os.path.join(root.common.test_dataset_root,
                                       "arial.ttf")),
               "weights_plotter": {"limit":
                                   get_config(root.weights_plotter.limit, 16)}
               }


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
    img[y:y + height, x: x + width] = numpy.array(
        bitmap.buffer, dtype=numpy.uint8).reshape(height, width)
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
        self.class_target.v = numpy.zeros(
            [10, 784], dtype=opencl_types.dtypes[root.common.dtype])
        for i in range(0, 10):
            img = do_plot(root.path_for_load_data,
                          "%d" % (i,), 28, 0.0, 1.0, 1.0, False, 28, 28)
            self.class_target.v[i] = img.ravel().astype(
                opencl_types.dtypes[root.common.dtype])
            formats.normalize(self.class_target.v[i])
        self.original_target = numpy.zeros(
            [self.original_labels.shape[0], self.class_target.v.shape[1]],
            dtype=opencl_types.dtypes[root.common.dtype])
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target.v[label]


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
            self, snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.class_samples = self.loader.class_samples

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.decision)
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.ev.y
        self.image_saver.target = self.ev.target
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size
        self.image_saver.this_time = self.decision.snapshot_time
        self.image_saver.gate_skip = ~self.decision.just_snapshotted

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(self, device=device)
        self.gd[-1].link_from(self.image_saver)
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

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        # Weights plotter
        # """
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = self.forward[0].input
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended
        # """
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
        self.plt_img.gate_block = ~self.decision.epoch_ended
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True


def run(load, main):
    load(Workflow, layers=root.layers_mnist784)
    main()
