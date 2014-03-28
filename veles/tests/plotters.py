"""
Created on Mar 17, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import filecmp
import matplotlib
matplotlib.use("cairo")
import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as pp
pp.ion()
import numpy
import os
import unittest

from veles.plotting_units import MatrixPlotter


class Test(unittest.TestCase):
    def add_ref(self, workflow):
        pass

    def testMatrixPlotter(self):
        mp = MatrixPlotter(self, name="Matrix")
        mp.input = numpy.asarray(
            [[[5753, 1, 22, 16, 10, 38, 32, 14, 30, 26],
              [0, 6559, 32, 27, 20, 24, 15, 38, 76, 23],
              [19, 35, 5554, 98, 36, 24, 21, 62, 39, 14],
              [5, 25, 67, 5600, 6, 132, 2, 23, 105, 72],
              [8, 6, 70, 7, 5472, 34, 41, 44, 28, 193],
              [44, 15, 16, 159, 7, 4953, 55, 10, 89, 40],
              [38, 7, 43, 17, 52, 87, 5716, 2, 41, 3],
              [6, 21, 60, 56, 13, 9, 3, 5941, 16, 124],
              [41, 59, 82, 105, 20, 69, 32, 13, 5355, 49],
              [9, 14, 12, 46, 206, 51, 1, 118, 72, 5405]]], order=2)
        mp.input_field = 0
        mp.cm = cm
        mp.lines = lines
        mp.patches = patches
        mp.pp = pp
        mp.show_figure = self.show_figure
        mp.redraw()
        tmp_file_name = "/tmp/MatrixPlotter.png"
        pp.savefig(tmp_file_name)
        self.assertTrue(filecmp.cmp(
            tmp_file_name,
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "res/MatrixPlotter.png")))

    def show_figure(self, figure):
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testMatrixPlotter']
    unittest.main()
