#!/usr/bin/python3.3 -O
"""
Created on Dec 13, 2013

Visualization of NN performance trained by gtzan.py.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
import logging
import matplotlib.pyplot as pp
import numpy
import pickle
import re
import sys


def evaluate_file(fnme, ff, window_size, shift_size, W, b):
    logging.info("Will test on: %s" % (fnme))

    labels = {"blues": 0,
              "country": 1,
              "jazz": 2,
              "pop": 3,
              "rock": 4,
              "classical": 5,
              "disco": 6,
              "hiphop": 7,
              "metal": 8,
              "reggae": 9}
    i_labels = {}
    for k, v in labels.items():
        i_labels[v] = k

    features = ["Energy", "Centroid", "Flux", "Rolloff", "ZeroCrossings"]

    norm_add = {'Rolloff': (-4194.1295584643221),
                'Centroid': (-2029.2263010288816),
                'ZeroCrossings': (-55.22063408843276),
                'Flux': (-0.91969947921678419),
                'Energy': (-10533447.118792284)}
    norm_mul = {'Rolloff': 0.00016505213410177407,
                'Centroid': 0.00014461928143403359,
                'ZeroCrossings': 0.0025266602711760356,
                'Flux': 0.066174679965212244,
                'Energy': 3.2792848503777384e-09}

    inp = numpy.zeros(len(features) * window_size, dtype=numpy.float64)

    lbl_re = re.compile("/(\w+)\.\w+\.\w+$")
    res = lbl_re.search(fnme)
    lbl = labels[res.group(1)]
    outs = numpy.zeros(len(labels), dtype=numpy.float64)

    limit = 2000000000
    for k in features:
        v = ff[k]["value"]
        limit = min(len(v), limit)
        v += norm_add[k]
        v *= norm_mul[k]

    x = numpy.arange(0, limit - window_size + 1, shift_size,
                     dtype=numpy.float64)
    x *= 0.01
    y = numpy.zeros([len(labels), len(x)], dtype=numpy.float64)
    yy = numpy.zeros([len(labels), len(x)], dtype=numpy.float64)
    i_shift = 0
    for offs in range(0, limit - window_size + 1, shift_size):
        offs2 = offs + window_size
        j = 0
        for k in features:
            v = ff[k]["value"]
            jj = j + window_size
            inp[j:jj] = v[offs:offs2]
            j = jj
        if inp.min() < -1:
            logging.info("input is out of range: %.6f" % (inp.min()))
        if inp.max() > 1:
            logging.info("input is out of range: %.6f" % (inp.max()))
        a = inp
        for i in range(len(W) - 1):
            weights = W[i]
            bias = b[i]
            out = numpy.dot(a, weights.transpose())
            out += bias
            out *= 0.6666
            numpy.tanh(out, out)
            out *= 1.7159
            a = out
        i = len(W) - 1
        weights = W[i]
        bias = b[i]
        out = numpy.dot(a, weights.transpose())
        out += bias
        # Apply softmax
        m = out.max()
        out -= m
        numpy.exp(out, out)
        smm = out.sum()
        out /= smm
        # Sum totals
        outs += out

        mx = numpy.argmax(out)
        logging.info(
            "%s: %s" % (i_labels[mx], " ".join("%.2f" % (x) for x in out)))

        y[0, i_shift] = out[0]
        yy[0, i_shift] = outs[0]
        for j in range(1, len(out)):
            y[j, i_shift] = out[j] + y[j - 1, i_shift]
            yy[j, i_shift] = outs[j] + yy[j - 1, i_shift]
        i_shift += 1

    logging.info("###########################################################")
    logging.info("Was tested on: %s" % (fnme))
    mx = numpy.argmax(outs)
    logging.info(
        "%s: %s" % (i_labels[mx], " ".join("%.2f" % (x) for x in outs)))
    if mx == lbl:
        logging.info("Recognized")
    else:
        logging.info("Not recognized")

    draw_plot("Points", x, y, i_labels, fnme)
    draw_plot("Incremental", x, yy, i_labels, fnme)

    pp.show()


def draw_plot(figure_label, x, y, i_labels, fnme):
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
    colors = ["blue",
              "pink",
              "green",
              "brown",
              "gold",
              "white",
              "red",
              "black",
              "gray",
              "orange"]

    fig = pp.figure(figure_label)
    ax = fig.add_subplot(111)
    #ax.set_ylim(0, 1)
    ax.set_title(fnme, fontsize=23)
    for i in range(len(y)):
        ax.plot(x, y[i], color=colors[i], label=i_labels[i], linewidth=4)
    ax.fill_between(x, y[0], 0, color=colors[0])
    for i in range(1, len(y)):
        ax.fill_between(x, y[i], y[i - 1], color=colors[i])

    ax.legend(ncol=3)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True,
                        help="Snapshot with trained network weights and bias.")
    parser.add_argument("-window_size", type=int, required=True,
                        help="Size of the scanning window.")
    parser.add_argument("-shift_size", type=int, required=True,
                        help="Size of the scanning window shift.")
    parser.add_argument("-input_file", type=str, required=True,
                        help="File on which visualization will be performed.")
    args = parser.parse_args()

    logging.info("Loading snapshot")
    fin = open(args.snapshot, "rb")
    (W, b) = pickle.load(fin)
    fin.close()
    logging.info("Done")

    logging.info("Loading dataset")
    fin = open("/data/veles/music/GTZAN/gtzan.pickle", "rb")
    data = pickle.load(fin)
    fin.close()
    logging.info("Done")

    if args.input_file in data["files"].keys():
        logging.info("Input file found in train dataset")
        features = data["files"][args.input_file]["features"]
    elif args.input_file in data["test"].keys():
        logging.info("Input files found in test dataset")
        features = data["test"][args.input_file]["features"]
    else:
        logging.info("Input file was not found in dataset")
        sys.exit(1)

    evaluate_file(args.input_file, features, args.window_size,
                  args.shift_size, W, b)


if __name__ == '__main__':
    main()
    sys.exit(0)
