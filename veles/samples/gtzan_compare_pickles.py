#!/usr/bin/python3.3 -O
"""
Created on Dec 20, 2013

Compares two gtzan.pickle's.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
import logging
import numpy
import pickle


def compare_features(av, bv, af, not_found, stats):
    for f in sorted(av.keys()):
        if f not in bv.keys():
            if f not in not_found:
                not_found.add(f)
                logging.info("Feature %s was not found "
                             "in the second pickle" % (f))
                stats["features_not_found"] += 1
            continue
        vle_a = av[f]["value"].ravel()
        vle_b = bv[f]["value"].ravel()
        if vle_a.size != vle_b.size:
            logging.info("Sizes of features %s differs in %s: %d vs %d" % (
                f, af, vle_a.size, vle_b.size))
            stats["sizes_differ"] += 1
            continue
        n = min(vle_a.size, vle_b.size)
        diff = numpy.max(numpy.fabs(vle_a[:n] - vle_b[:n]))
        if diff > 0.000001:
            logging.info("Features %s differs by %.6f in %s" % (f, diff, af))
            stats["n_fail"] += 1
        else:
            stats["features_ok"] += 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("A", type=str, help="First gtzan.pickle")
    parser.add_argument("B", type=str, help="Second gtzan.pickle")
    args = parser.parse_args()

    logging.info("Loading %s" % (args.A))
    fin = open(args.A, "rb")
    a = pickle.load(fin)
    fin.close()
    logging.info("Done")

    logging.info("Loading %s" % (args.B))
    fin = open(args.B, "rb")
    b = pickle.load(fin)
    fin.close()
    logging.info("Done")

    not_f = set()
    stat = {"n_fail": 0,
            "features_ok": 0,
            "files_differ": 0,
            "sizes_differ": 0,
            "features_not_found": 0}

    for af_ in sorted(a["files"].keys()):
        if af_ not in b["files"].keys():
            stat["files_differ"] += 1
            logging.info("%s was not found in %s[files]" % (af_, args.B))
            if "test" not in b.keys():
                continue
            if af_ not in b["test"].keys():
                logging.info("%s was not found in %s[test]" % (af_, args.B))
                continue
            bv_ = b["test"][af_]["features"]
        else:
            bv_ = b["files"][af_]["features"]
        av_ = a["files"][af_]["features"]
        compare_features(av_, bv_, af_, not_f, stat)

    if "test" in a.keys():
        for af_ in sorted(a["test"].keys()):
            if "test" not in b.keys() or af_ not in b["test"].keys():
                stat["files_differ"] += 1
                if "test" in b.keys():
                    logging.info("%s was not found in %s[test]"
                                 % (af_, args.B))
                if af_ not in b["files"].keys():
                    logging.info(
                        "%s was not found in %s[files]" % (af_, args.B))
                    continue
                bv_ = b["files"][af_]["features"]
            else:
                bv_ = b["test"][af_]["features"]
            av_ = a["test"][af_]["features"]
            compare_features(av_, bv_, af_, not_f, stat)

    logging.info(str(stat))
