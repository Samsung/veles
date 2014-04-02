#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for function approximation.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import scipy.io

from veles.config import root, get_config
import veles.error as error
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


root.update = {"decision": {"fail_iterations":
                            get_config(root.decision.fail_iterations, 1000),
                            "snapshot_prefix":
                            get_config(root.decision.snapshot_prefix,
                                       "approximator"),
                            "store_samples_mse":
                            get_config(root.decision.store_samples_mse, True)},
               "global_alpha": get_config(root.global_alpha, 0.01),
               "global_lambda": get_config(root.global_lambda, 0.00005),
               "layers": get_config(root.layers, [810, 9]),
               "loader": {"minibatch_maxsize":
                          get_config(root.loader.minibatch_maxsize, 81)},
               "path_for_target_data":
               get_config(root.path_for_target_data,
                          ["/data/veles/approximator/all_org_appertures.mat"]),
               "path_for_train_data":
               get_config(root.path_for_train_data,
                          ["/data/veles/approximator/all_dec_appertures.mat"])
               }


class Loader(loader.ImageLoader):
    def load_original(self, fnme):
        a = scipy.io.loadmat(fnme)
        for key in a.keys():
            if key[0] != "_":
                vle = a[key]
                break
        else:
            raise error.ErrBadFormat("Could not find variable to import "
                                     "in %s" % (fnme))
        aa = numpy.zeros(vle.shape, dtype=opencl_types.dtypes[
            root.common.dtype])
        aa[:] = vle[:]
        return (aa, [])

    def load_data(self):
        super(Loader, self).load_data()
        return
        if self.class_samples[1] == 0:
            n = self.class_samples[2] * 10 // 70
            self.class_samples[1] = n
            self.class_samples[2] -= n

    def initialize(self):
        super(Loader, self).initialize()
        self.shuffle_validation_train()
        self.info("data range: (%.6f, %.6f), target range: (%.6f, %.6f)"
                  % (self.original_data.min(), self.original_data.max(),
                     self.original_target.min(), self.original_target.max()))
        # Normalization
        for i in range(0, self.original_data.shape[0]):
            data = self.original_data[i]
            data /= 127.5
            data -= 1.0
            m = data.mean()
            data -= m
            data *= 0.5
            target = self.original_target[i]
            target /= 127.5
            target -= 1.0
            target -= m
            target *= 0.5

        self.info("norm data range: (%.6f, %.6f), "
                  "norm target range: (%.6f, %.6f)"
                  % (self.original_data.min(), self.original_data.max(),
                     self.original_target.min(), self.original_target.max()))
        """
        train_data = self.original_data[self.nextclass_offs[1]:
                                        self.nextclass_offs[2]]
        train_target = self.original_target[self.nextclass_offs[1]:
                                            self.nextclass_offs[2]]

        self.data_IMul, self.data_IAdd = formats.normalize_pointwise(
                                                            train_data)
        self.target_IMul, self.target_IAdd = formats.normalize_pointwise(
                                                            train_target)

        train_data *= self.data_IMul
        train_data += self.data_IAdd
        train_target *= self.target_IMul
        train_target += self.target_IAdd

        train_data = self.original_data[self.nextclass_offs[1]:
                                        self.nextclass_offs[2]]
        train_target = self.original_target[self.nextclass_offs[1]:
                                            self.nextclass_offs[2]]

        self.info("train data normed range: (%.6f, %.6f)" % (
            train_data.min(), train_data.max()))
        self.info("train target normed range: (%.6f, %.6f)" % (
            train_target.min(), train_target.max()))

        if self.class_samples[0]:
            test_data = self.original_data[:self.nextclass_offs[0]]
            formats.assert_addr(test_data, self.original_data)
            test_target = self.original_target[:self.nextclass_offs[0]]
            formats.assert_addr(test_target, self.original_target)

            test_data *= self.data_IMul
            test_data += self.data_IAdd
            test_target *= self.target_IMul
            test_target += self.target_IAdd

            self.info("test data normed range: (%.6f, %.6f)" % (
                test_data.min(), test_data.max()))
            self.info("test target normed range: (%.6f, %.6f)" % (
                test_target.min(), test_target.max()))

        if self.class_samples[1]:
            validation_data = self.original_data[self.nextclass_offs[0]:
                                                 self.nextclass_offs[1]]
            formats.assert_addr(validation_data, self.original_data)
            validation_target = self.original_target[self.nextclass_offs[0]:
                                                     self.nextclass_offs[1]]
            formats.assert_addr(validation_target, self.original_target)

            validation_data *= self.data_IMul
            validation_data += self.data_IAdd
            validation_target *= self.target_IMul
            validation_target += self.target_IAdd

            self.info("validation data normed range: (%.6f, %.6f)" % (
                validation_data.min(), validation_data.max()))
            self.info("validation target normed range: (%.6f, %.6f)" % (
                validation_target.min(), validation_target.max()))
        """


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

        self.loader = Loader(
            self, train_paths=root.path_for_train_data,
            target_paths=root.path_for_target_data,
            minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
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
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            store_samples_mse=root.decision.store_samples_mse,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
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

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Average plotter
        self.plt_avg = []
        styles = ["", "b-", "k-"]  # ["r-", "b-", "k-"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_avg.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_avg[-1].input = self.decision.epoch_metrics
            self.plt_avg[-1].input_field = i
            self.plt_avg[-1].link_from(self.plt_avg[-2] if j
                                       else self.decision)
            self.plt_avg[-1].gate_block = (Bool(False) if j
                                           else ~self.decision.epoch_ended)
            j += 1
        self.plt_avg[0].clear_plot = True
        # Max plotter
        self.plt_max = []
        styles = ["", "b--", "k--"]  # ["r--", "b--", "k--"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt_max[-2] if j
                                       else self.plt_avg[-1])
            j += 1
        # Min plotter
        self.plt_min = []
        styles = ["", "b:", "k:"]  # ["r:", "b:", "k:"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_min[-2] if j
                                       else self.plt_max[-1])
            j += 1
        self.plt_min[-1].redraw_plot = True
        # Histogram plotter
        self.plt_hist = plotting_units.MSEHistogram(self, name="Histogram")
        self.plt_hist.link_from(self.decision)
        self.plt_hist.mse = self.decision.epoch_samples_mse[2]
        self.plt_hist.gate_block = ~self.decision.epoch_ended
        # Plot
        self.plt = plotting_units.Plot(self, name="Plot", ylim=[-1.1, 1.1])
        del self.plt.inputs[:]
        self.plt.inputs.append(self.loader.minibatch_data)
        self.plt.inputs.append(self.loader.minibatch_target)
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.plt.inputs.append(self.forward[-1].output)
        del self.plt.input_fields[:]
        self.plt.input_fields.append(0)
        self.plt.input_fields.append(0)
        self.plt.input_fields.append(0)
        del self.plt.input_styles[:]
        self.plt.input_styles.append("k-")
        self.plt.input_styles.append("g-")
        self.plt.input_styles.append("b-")
        self.plt.link_from(self.decision)
        self.plt.gate_block = ~self.decision.epoch_ended


def run(load, main):
    load(Workflow, layers=root.layers)
    main()
