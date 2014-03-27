"""
Created on Apr 19, 2013
model use always dataset in the txt format

(data parameters is  in the  separate configuration file data.
 need to write a module configuration file read data
 need to write a reader txt files in any format.
)

Example wine data set -13 paramets input.  3 class outputs / 178 samples (all
dataset and train and test)
Fixed structure NN (5[Tanh]-3[SoftMax])

All train set is in the train-batch BP methods
(simple)  (type of operation (surgery or criterion) must be specified
in the configuration file jobs)

Criterion of learning is to remember all. (-"-)

Result is in the serialize (-"-)

no testing for test data set. (-"-)

@author: Seresov Denis <d.seresov@samsung.com>
"""


import logging
import numpy

import veles.opencl as opencl
import veles.plotting_units as plotting_units
import veles.units as units
import veles.znicz.all2all as all2all
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


def strf(x):
    return "%.4f" % (x)


class model_WF_wine(units.Pickleable):
    """UUseCaseTxt.

    Attributes:
        start_point: Filter.
        end_point: EndPoint.
        t: t.
    """
    def __init__(self, data_set=None, param=None, cpu=True):
        super(model_WF_wine, self).__init__()
        if data_set is None:
            data_set = {}
        self.data_set = data_set
        if param is None:
            param = {}
        self.param = param

        # rnd.default.seed(numpy.fromfile("seed", numpy.integer, 1024))
        # TODO(d.seresov): get it from config service
        numpy.random.seed(numpy.fromfile("scripts/seed", numpy.integer))
        dev = None
        if not cpu:
            dev = opencl.Device()

        # Setup notification flow
        self.start_point = units.Unit()

        # self.debug(self.config_data_seta)
        # self.debug(self.config_datasa)
        # self.debug(self.data_set)
        t = loader.Loader(self.data_set, self.param)
        self.t = t
        # sys.exit()
        self.debug("1")
        self.debug(t)
        t.link_from(self.start_point)
        self.debug("2")

        self.rpt.link_from(self.start_point)

        aa1 = all2all.All2AllTanh(output_shape=[70], device=dev)
        aa1.input = t.output2
        aa1.link_from(self.rpt)

        out = all2all.All2AllSoftmax(output_shape=[3], device=dev)
        out.input = aa1.output
        out.link_from(aa1)

        ev = evaluator.EvaluatorSoftmax(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.params = t.params
        ev.TrainIndex = t.TrainIndex
        ev.ValidIndex = t.ValidIndex
        ev.TestIndex = t.TestIndex
        # ev.Index=t.Index
        ev.link_from(out)

        plt_ok_train = plotting_units.AccumulatingPlotter(
            device=dev, name="train")
        plt_ok_train.input = ev.status
        plt_ok_train.input_field = 'n_ok'
        plt_ok_train.link_from(ev)

        plt_total_train = plotting_units.AccumulatingPlotter(
            device=dev, name="train", plot_style="blue")
        plt_total_train.input = ev.status
        plt_total_train.input_field = 'count_train'
        plt_total_train.link_from(ev)

        plt_ok_valid = plotting_units.AccumulatingPlotter(
            device=dev, name="validation")
        plt_ok_valid.input = ev.status
        plt_ok_valid.input_field = 'n_ok_v'
        plt_ok_valid.link_from(ev)

        plt_total_valid = plotting_units.AccumulatingPlotter(
            device=dev, name="validation", plot_style="blue")
        plt_total_valid.input = ev.status
        plt_total_valid.input_field = 'count_valid'
        plt_total_valid.link_from(ev)

        """
        ev = evaluator.EvaluatorMSE(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.params=t.params
        ev.TrainIndex = t.TrainIndex
        ev.ValidIndex = t.ValidIndex
        ev.TestIndex  = t.TestIndex
        ev.Index=t.Index
        ev.link_from(out)
        """

        gdsm = gd.GDSM(self, device=dev)
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.L = ev.L
        gdsm.err_y = ev.err_y

        gd1 = gd.GDTanh(self, device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gdsm.err_h
        gd1.L = ev.L
        gd1.link_from(gdsm)

        self.rpt.link_from(gd1)
        self.end_point.link_from(ev)
        gdsm.link_from(self.end_point)

        self.ev = ev
        self.sm = out  # ?
        self.gdsm = gdsm  # ?
        self.gd1 = gd1  # ?
        self.debug("ok_init ")

    def do_log(self, out, gdsm, gd1):
        return
        flog = open("logs/out.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes))
        flog.write("\nSoftMax layer input:\n")
        for sample in out.input.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer output:\n")
        for sample in out.output.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer weights:\n")
        for sample in out.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer bias:\n")
        flog.write(" ".join(strf(x) for x in out.bias.v))
        flog.write("\n(min, max)(input, output, weights, bias) = "
                   "((%f, %f), (%f, %f), (%f, %f), (%f, %f)\n" %
                   (out.input.batch.min(), out.input.batch.max(),
                    out.output.batch.min(), out.output.batch.max(),
                    out.weights.v.min(), out.weights.v.max(),
                    out.bias.v.min(), out.bias.v.max()))
        flog.write("\n")
        flog.close()

        flog = open("logs/gdsm.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes))
        flog.write("\nGD SoftMax err_y:\n")
        for sample in gdsm.err_y.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax err_h:\n")
        for sample in gdsm.err_h.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax weights:\n")
        for sample in gdsm.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax bias:\n")
        flog.write(" ".join(strf(x) for x in gdsm.bias.v))
        flog.write("\n(min, max)(err_y, err_h, weights, bias) = ((%f, %f), "
                   "(%f, %f), (%f, %f), (%f, %f)\n" %
                   (gdsm.err_y.batch.min(), gdsm.err_y.batch.max(),
                    gdsm.err_h.batch.min(), gdsm.err_h.batch.max(),
                    gdsm.weights.v.min(), gdsm.weights.v.max(),
                    gdsm.bias.v.min(), gdsm.bias.v.max()))
        flog.write("\n")
        flog.close()

        flog = open("logs/gd1.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes))
        flog.write("\nGD1 err_y:\n")
        for sample in gd1.err_y.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 err_h:\n")
        for sample in gd1.err_h.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 weights:\n")
        for sample in gd1.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 bias:\n")
        flog.write(" ".join(strf(x) for x in gd1.bias.v))
        flog.write("\n(min, max)(err_y, err_h, weights, bias) = ((%f, %f), "
                   "(%f, %f), (%f, %f), (%f, %f)\n" %
                   (gd1.err_y.batch.min(), gd1.err_y.batch.max(),
                    gd1.err_h.batch.min(), gd1.err_h.batch.max(),
                    gd1.weights.v.min(), gd1.weights.v.max(),
                    gd1.bias.v.min(), gd1.bias.v.max()))
        flog.write("\n")
        flog.close()

    def run(self):
        # Assume that train_param already exists at this point
        _t = self.param['train_param']
        logging.debug(_t)
        logging.debug(" WF WINE RUN START")
        # Start the process
        self.sm.threshold = _t['threshold']
        self.sm.threshold_low = _t['threshold_low']
        self.ev.threshold = _t['threshold']
        self.ev.threshold_low = _t['threshold_low']
        self.gdsm.global_alpha = _t['global_alpha']
        self.gdsm.global_lambda = _t['global_lambda']
        self.gd1.global_alpha = _t['global_alpha']
        self.gd1.global_lambda = _t['global_lambda']
        logging.info()
        logging.info("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        # for l in self.t.labels.batch:
        #    self.debug(l)
        # sys.exit()
        logging.info()
        logging.info("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()
