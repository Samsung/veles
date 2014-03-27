#!/usr/bin/python3.3
"""
Created on Sep 2, 2013

Will normalize counts of *.jp2 in the supplied folder
by replicating some of the found files.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import argparse
import numpy
import os
import re
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir", type=str, required=True,
        help="Directory with channels")
    parser.add_argument(
        "-at_least", type=int, required=True,
        help="Minimum number of *.jp2 in each subfolder")
    parser.add_argument(
        "-seed", type=str, required=True,
        help="File with seed for choosing of file to replicate")
    args = parser.parse_args()

    numpy.random.seed(numpy.fromfile(args.seed, dtype=numpy.int32, count=1024))

    first = True
    jp2 = re.compile("\.jp2$", re.IGNORECASE)
    for basedir, dirlist, filelist in os.walk(args.dir, topdown=False):
        found_files = []
        for nme in filelist:
            if jp2.search(nme) is not None:
                found_files.append("%s/%s" % (basedir, nme))
        n = len(found_files)
        if n >= args.at_least or n == 0:
            continue
        print("Will replicate some of %d files in %s up to %d" % (
            n, basedir, args.at_least))
        if first:
            first = False
            k = 15
            for kk in range(k, 0, -1):
                print("Will do the rest after %d seconds" % (kk))
                time.sleep(1)
            print("Will replicate now")
        for i in range(args.at_least - n):
            ii = numpy.random.randint(n)
            nme = found_files[ii]
            shutil.copy(nme, "%s_%d.jp2" % (nme[:-4], i))

    print("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
