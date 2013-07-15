#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

Extract and scale y, u, v to png from raw uyvy 1920*1080 files.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import argparse
import numpy
import scipy.misc
import scipy.ndimage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="input .raw file", required=True)
    parser.add_argument("-d", type=str, help="output directory", required=True)
    parser.add_argument("-x", type=int, help="rect.x", required=True)
    parser.add_argument("-y", type=int, help="rect.y", required=True)
    parser.add_argument("-width", type=int, help="rect.width", required=True)
    parser.add_argument("-height", type=int, help="rect.height", required=True)
    parser.add_argument("-scale", type=float, help="scale factor",
                        required=True)
    parser.add_argument("-gray", type=bool, help="grayscale", default=False)
    args = parser.parse_args()

    y = numpy.zeros([args.height, args.width], dtype=numpy.uint8)
    u = numpy.zeros([args.height, args.width >> 1], dtype=numpy.uint8)
    v = numpy.zeros([args.height, args.width >> 1], dtype=numpy.uint8)

    aw = int(numpy.round(args.width * args.scale))
    ah = int(numpy.round(args.height * args.scale))
    if args.gray:
        a = numpy.zeros([ah, aw], dtype=numpy.uint8)
    else:
        a = numpy.zeros([ah << 1, aw], dtype=numpy.uint8)

    fin = open(args.i, "rb")
    ii = 0
    while True:
        ii += 1
        print(ii)
        try:
            img = numpy.fromfile(fin, numpy.uint8, 1920 // 2 * 1080 * 4).\
                reshape(1080, 1920 // 2, 4)
            img = img[args.y:args.y + args.height,
                      args.x // 2:(args.x + args.width) // 2]
        except ValueError:
            break
        for row in range(0, img.shape[0]):
            for col in range(0, img.shape[1]):
                pix = img[row, col]
                u[row, col] = pix[0]
                v[row, col] = pix[2]
                y[row, col << 1] = pix[1]
                y[row, (col << 1) + 1] = pix[3]

        if args.scale != 1.0:
            ay = scipy.ndimage.zoom(y, args.scale, order=1)
            au = scipy.ndimage.zoom(u, args.scale, order=1)
            av = scipy.ndimage.zoom(v, args.scale, order=1)
        else:
            ay = y
            au = u
            av = v

        a[:ah, :] = ay[:]
        if not args.gray:
            a[ah:, :aw >> 1] = au
            a[ah:, aw >> 1:] = av

        scipy.misc.imsave("%s/%04d.png" % (args.d, ii), a)

    print("Done")
