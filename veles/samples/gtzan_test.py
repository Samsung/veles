#!/usr/bin/python3.3 -O
"""
Created on Dec 11, 2013

Test performance of NN trained by gtzan.py.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
import logging
import numpy
import pickle
import re
import sys


def evaluate_dataset(dataset, window_size, W, b):
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

    features = ["Energy", "Centroid", "Flux", "Rolloff",
                "ZeroCrossings"]

    norm_add = {'Centroid': (-2029.2262731600895),
                'Energy': (-10533446.715802385),
                'Flux': (-0.91969949785961735),
                'Rolloff': (-4194.1299697454906),
                'ZeroCrossings': (-55.22063408843276)}
    norm_mul = {'Centroid': 0.00014461928085116515,
                'Energy': 3.2792848460441024e-09,
                'Flux': 0.066174680046850856,
                'Rolloff': 0.00016505214530598153,
                'ZeroCrossings': 0.0025266602711760356}

    inp = numpy.zeros(len(features) * window_size, dtype=numpy.float64)

    n_ok = 0
    n_fail = 0

    lbl_re = re.compile("/(\w+)\.\w+\.\w+$")
    outs = numpy.zeros(len(labels), dtype=numpy.float64)
    for fnme, fvle in dataset.items():
        limit = 2000000000
        ff = fvle["features"]
        res = lbl_re.search(fnme)
        lbl = labels[res.group(1)]
        outs[:] = 0
        for k in features:
            v = ff[k]["value"]
            ff[k]["value"] = v.astype(numpy.float64)
            v = ff[k]["value"]
            limit = min(len(v), limit)
            v += norm_add[k]
            v *= norm_mul[k]
        for offs in range(0, limit - window_size + 1, window_size // 10):
            offs2 = offs + window_size
            j = 0
            for k in features:
                v = ff[k]["value"]
                jj = j + window_size
                inp[j:jj] = v[offs:offs2]
                j = jj
            if inp.min() < -1.0001:
                logging.info("input is out of range: %.6f" % (inp.min()))
            if inp.max() > 1.0001:
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
        mx = numpy.argmax(outs)
        if mx == lbl:
            n_ok += 1
        else:
            n_fail += 1
            logging.info("FAIL: %s as %s / %s" %
                         (fnme, i_labels[mx],
                          " ".join("%.2f" % (x) for x in outs)))

    logging.info("n_ok: %d" % (n_ok))
    logging.info("n_fail: %d" % (n_fail))
    logging.info("%.2f%% errors" % (100.0 * n_fail / (n_ok + n_fail)))

    return n_ok, n_fail


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-pickle", type=str,
                        default="/data/veles/music/GTZAN/gtzan.pickle",
                        help="Snapshot with trained network weights and bias.")
    parser.add_argument("-snapshot", type=str, required=True,
                        help="Snapshot with trained network weights and bias.")
    parser.add_argument("-window_size", type=int, required=True,
                        help="Size of the scanning window.")
    args = parser.parse_args()

    logging.info("Loading snapshot")
    fin = open(args.snapshot, "rb")
    (W, b) = pickle.load(fin)
    fin.close()
    logging.info("Done")

    logging.info("Loading dataset")
    fin = open(args.pickle, "rb")
    data = pickle.load(fin)
    fin.close()
    logging.info("Done")

    logging.info("Will evaluate test dataset")
    n_ok_test, n_fail_test = evaluate_dataset(data["test"],
                                              args.window_size, W, b)
    logging.info("Will evaluate train dataset")
    n_ok_train, n_fail_train = evaluate_dataset(data["files"],
                                                args.window_size, W, b)

    logging.info("###################################################")
    logging.info("Test: %d errors (%.2f%%)" %
                 (n_fail_test, 100.0 * n_fail_test / (n_ok_test +
                                                      n_fail_test)))
    logging.info("Train: %d errors (%.2f%%)" %
                 (n_fail_train, 100.0 * n_fail_train / (n_ok_train +
                                                        n_fail_train)))

if __name__ == '__main__':
    main()
    sys.exit(0)
