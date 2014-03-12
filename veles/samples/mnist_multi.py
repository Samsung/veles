#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import numpy
import os
import sys

import veles.config as config
import veles.opencl as opencl
import veles.rnd as rnd
import veles.samples.mnist as mnist
import veles.samples.mnist784 as mnist784


def main():
    #if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    logging.basicConfig(level=logging.INFO)

    rnd.default.seed(numpy.fromfile("%s/seed" % (os.path.dirname(__file__)),
                                    numpy.int32, 1024))
    rnd.default2.seed(numpy.fromfile("%s/seed2" % (os.path.dirname(__file__)),
                                     numpy.int32, 1024))
    device = opencl.Device()

    logging.info("Will execute SoftMax workflow")
    w = mnist.Workflow(None, layers=[100, 10], device=device)
    w.initialize(device=device, global_alpha=0.1, global_lambda=0.0)
    w.run()

    logging.info("Will execute MSE workflow")
    w0 = w
    w = mnist784.Workflow(None, layers=[100, 784, 784], device=device)
    w.initialize(global_alpha=0.001, global_lambda=0.00005)
    w0.forward[0].weights.map_read()
    w0.forward[0].bias.map_read()
    w.forward[0].weights.map_invalidate()
    w.forward[0].bias.map_invalidate()
    w.forward[0].weights.v[:] = w0.forward[0].weights.v[:]
    w.forward[0].bias.v[:] = w0.forward[0].bias.v[:]
    w.run()

    logging.debug("End of job")


if __name__ == "__main__":
    main()
    if config.plotters_disabled:
        os._exit(0)
    sys.exit(0)
