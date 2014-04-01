#!/usr/bin/python3.3
"""
Created on Dec 20, 2013.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import logging
import matplotlib.pyplot as pp
import numpy
import pickle
import os
import sys

from veles.config import root, get_config
from libSoundFeatureExtraction.python.sound_feature_extraction import (
    features_xml)
import veles.audio_file_loader as audio_file_loader
import veles.launcher as launcher
import veles.opencl as opencl
import veles.snd_features as snd_features
import veles.units as units
import veles.workflows as workflows

gtzan_dir = os.path.join(root.common.test_dataset_root, "music/GTZAN")
music_dir = os.path.join(root.common.test_dataset_root, "music")

root.labels = get_config(root.labels, {"blues": 0,
                                       "country": 1,
                                       "jazz": 2,
                                       "pop": 3,
                                       "rock": 4,
                                       "classical": 5,
                                       "disco": 6,
                                       "hiphop": 7,
                                       "metal": 8,
                                       "reggae": 9})

root.norm_add = get_config(root.norm_add,
                           {'Rolloff': (-4194.1299697454906),
                            'Centroid': (-2029.2262731600895),
                            'ZeroCrossings': (-55.22063408843276),
                            'Flux': (-0.91969949785961735),
                            'Energy': (-10533446.715802385)})
root.norm_mul = get_config(root.norm_mul,
                           {'Rolloff': 0.00016505214530598153,
                            'Centroid': 0.00014461928085116515,
                            'ZeroCrossings': 0.0025266602711760356,
                            'Flux': 0.066174680046850856,
                            'Energy': 3.2792848460441024e-09})

root.update = {"colors": get_config(root.colors, ["blue",
                                                  "pink",
                                                  "green",
                                                  "brown",
                                                  "gold",
                                                  "white",
                                                  "red",
                                                  "black",
                                                  "gray",
                                                  "orange"]),
               "features":
               get_config(root.features,
                          ["Energy", "Centroid", "Flux", "Rolloff",
                           "ZeroCrossings"]),
               "file_name":
               get_config(root.file_name,
                          os.path.join(music_dir, "22.flac")),
               "graphics": get_config(root.graphics, 1),
               "limit": get_config(root.limit, 2000000000),
               "path_for_features":
               get_config(root.path_for_features,
                          os.path.join(music_dir, "features.xml")),
               "plotter_window_name": get_config(root.plotter_window_name, ""),
               "shift_size": get_config(root.shift_size, 10),
               "snapshot_forward":
               get_config(root.snapshot_forward,
                          os.path.join(gtzan_dir,
                                       "gtzan_1000_500_10_28.88pt_Wb.pickle")),
               "title_fontsize": get_config(root.title_fontsize, 23),
               "window_size": get_config(root.window_size, 100)
               }


class Workflow(workflows.OpenCLWorkflow):
    """Workflow.
    """
    def __init__(self, workflow, **kwargs):

        device = kwargs.get("device")
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.audio_loader = audio_file_loader.AudioFileLoader()
        self.audio_loader.link_from(self.start_point)

        self.extr = snd_features.SoundFeatures(None)
        self.extr.link_from(self.audio_loader)
        self.extr.inputs = self.audio_loader.outputs

        self.forward = Forward()
        self.forward.ff = self.extr
        self.forward.ff_key = "outputs"
        self.forward.link_from(self.extr)

        self.end_point.link_from(self.forward)

    def initialize(self, file, feature_file, W, b,
                   window_size, shift_size):
        self.audio_loader.files_list = [file]
        self.forward.W = W
        self.forward.b = b
        self.forward.window_size = window_size
        self.forward.shift_size = shift_size
        features = features_xml.FeaturesXml.parse(feature_file)
        self.extr.add_features(features)
        return super(Workflow, self).initialize()


class Forward(units.Unit):
    def __init__(self, ff=None, W=None, b=None, window_size=None,
                 shift_size=None):
        super(Forward, self).__init__()
        self.W = W
        self.b = b
        self.window_size = window_size
        self.ff = ff
        self.shift_size = shift_size
        self.outs = None

    def run(self):
        ff = self.ff.__dict__[self.ff_key]
        ff = ff[0][0]

        labels = root.labels
        self.i_labels = {}
        for k, v in labels.items():
            self.i_labels[v] = k
        features = root.features
        norm_add = root.norm_add
        norm_mul = root.norm_mul
        limit = root.limit
        for k in features:
            v = ff[k]
            limit = min(len(v), limit)

        inp = numpy.zeros(len(features) * self.window_size,
                          dtype=numpy.float64)
        self.x = numpy.arange(
            0, limit - self.window_size + 1, self.shift_size,
            dtype=numpy.float64)
        self.x *= 0.01
        self.y = numpy.zeros([len(labels), len(self.x)], dtype=numpy.float64)
        self.yy = numpy.zeros([len(labels), len(self.x)], dtype=numpy.float64)
        i_shift = 0
        self.outs = numpy.zeros(len(labels), dtype=numpy.float64)
        self.outs_index = numpy.zeros(len(labels), dtype=numpy.float64)
        window_offs = 0
        eod = False
        while not eod:
            jj = 0
            offs2 = window_offs + self.window_size
            for k in features:
                v = ff[k]
                if window_offs + self.window_size > len(v):
                    eod = True
                    break
                j = jj
                jj = j + self.window_size
                inp[j:jj] = v[window_offs:offs2]
                inp[j:jj] += norm_add[k]
                inp[j:jj] *= norm_mul[k]
            if eod:
                break
            window_offs += self.shift_size
            if inp.min() < -1.0001:
                logging.info("input is out of range: %.6f" % (inp.min()))
            if inp.max() > 1.0001:
                logging.info("input is out of range: %.6f" % (inp.max()))
            a = inp
            for i in range(len(self.W) - 1):
                weights = self.W[i]
                bias = self.b[i]
                out = numpy.dot(a, weights.transpose())
                out += bias
                out *= 0.6666
                numpy.tanh(out, out)
                out *= 1.7159
                a = out
            i = len(self.W) - 1
            weights = self.W[i]
            bias = self.b[i]
            out = numpy.dot(a, weights.transpose())
            out += bias
            # Apply softmax
            m = out.max()
            out -= m
            numpy.exp(out, out)
            smm = out.sum()
            out /= smm
            # Sum totals
            self.outs += out
            logging.info("Out: %s" % (out))
            self.y[0, i_shift] = out[0]
            self.yy[0, i_shift] = self.outs[0]
            for j in range(1, len(out)):
                self.y[j, i_shift] = out[j] + self.y[j - 1, i_shift]
                self.yy[j, i_shift] = self.outs[j] + self.yy[j - 1, i_shift]
            i_shift += 1

        logging.info("Out_final: %s" % (self.outs))
        # self.outs_index = self.outs
        self.outs_index = self.outs.argsort()
        genre = self.i_labels[self.outs_index[9]]
        procent = self.outs[self.outs_index[9]]
        logging.info("Best genre: %s (%s)" % (genre, procent))
        logging.info(
            "Best 3 genre: %s (%s), %s (%s), %s (%s)" %
            (self.i_labels[self.outs_index[9]], self.outs[self.outs_index[9]],
             self.i_labels[self.outs_index[8]], self.outs[self.outs_index[8]],
             self.i_labels[self.outs_index[7]], self.outs[self.outs_index[7]]))


def draw_plot(figure_label, x, y, i_labels, fnme, name, left_legend=False):
    """
        "blues": 0,
        "country": 1,
        "jazz": 2,
        "pop": 3,
        "rock": 4,
        "classical": 5,
        "disco": 6,
        "hiphop": 7,
        "metal": 8,
        "reggae": 9
    """
    colors = root.colors

    fig = pp.figure(figure_label)
    ax = fig.add_subplot(111)
    # ax.set_ylim(0, 1)
    ax.set_title(name if len(name) else fnme, fontsize=root.title_fontsize)
    for i in range(len(y)):
        ax.plot(x, y[i], color=colors[i], label=i_labels[i], linewidth=4)
    ax.fill_between(x, y[0], 0, color=colors[0])
    for i in range(1, len(y)):
        ax.fill_between(x, y[i], y[i - 1], color=colors[i])

    if left_legend:
        ax.legend(ncol=3, loc=2)
    else:
        ax.legend(ncol=3)


def main():
    l = launcher.Launcher()
    fin = open(root.snapshot_forward, "rb")
    W, b = pickle.load(fin)
    fin.close()
    device = None if l.is_master else opencl.Device()
    w = Workflow(l, device=device)
    w.initialize(file=root.file_name, feature_file=root.path_for_features,
                 W=W, b=b, window_size=root.window_size,
                 shift_size=root.shift_size)
    l.run()

    if root.graphics:
        draw_plot("Points", w.forward.x, w.forward.y, w.forward.i_labels,
                  root.file_name, root.plotter_window_name)
        draw_plot("Incremental", w.forward.x, w.forward.yy, w.forward.i_labels,
                  root.file_name, root.plotter_window_name, True)

        pp.show()


if __name__ == "__main__":
    main()
    sys.exit(0)
