#!/usr/bin/python3.3 -O
"""
Created on Dec 12, 2013

Splits gtzan.pickle in training and test data.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import pickle
import re


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    fin = open("/data/veles/music/GTZAN/gtzan_raw.pickle", "rb")
    data = pickle.load(fin)
    fin.close()
    test = {}
    train = data["files"]
    try:
        fin = open("/data/veles/music/GTZAN/test.txt", "r")
        logging.info("Found saved test.txt")
        fnmes = fin.readlines()
        fin.close()
        for fnme in fnmes:
            test[fnme[:-1]] = train.pop(fnme[:-1])
    except:
        fnmes = list(train.keys())
        n_files = len(train)
        numpy.random.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
        lbl_re = re.compile("/(\w+)\.\w+\.\w+$")
        lbls = {}
        for fnme in fnmes:
            res = lbl_re.search(fnme)
            lbl = res.group(1)
            lbls[lbl] = lbls.get(lbl, 0) + 1
        n_total = 0
        for lbl in lbls.keys():
            n = lbls[lbl] // 10
            lbls[lbl] = n
            n_total += n
        top = len(fnmes)
        while n_total > 0 and top > 0:
            n = numpy.random.randint(top)
            fnme = fnmes[n]
            res = lbl_re.search(fnme)
            lbl = res.group(1)
            if lbls[lbl] <= 0:
                top -= 1
                t = fnmes[n]
                fnmes[n] = fnmes[top]
                fnmes[top] = t
                continue
            lbls[lbl] -= 1
            n_total -= 1
            test[fnme] = train.pop(fnme)
            fnmes.pop(n)
            top -= 1

    data["test"] = test
    logging.info("Extracted %d test samples" % (len(test)))
    fout = open("/data/veles/music/GTZAN/gtzan.pickle", "wb")
    pickle.dump(data, fout)
    fout.close()
