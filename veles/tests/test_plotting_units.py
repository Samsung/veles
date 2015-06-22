# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 17, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import io
import logging
import matplotlib
from veles.compat import from_none

matplotlib.use("cairo")
import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as pp
pp.ion()
import numpy
import os
import pickle
from PIL import Image
import tempfile
import unittest

from veles.plotting_units import AccumulatingPlotter, MatrixPlotter, \
    ImagePlotter, ImmediatePlotter, Histogram


class Test(unittest.TestCase):
    def add_ref(self, workflow):
        pass

    def show_figure(self, figure):
        pass

    def run_plotter(self, plotter, save_on_disk=True):
        plotter.stripped_pickle = True
        plotter = pickle.loads(pickle.dumps(plotter))
        plotter.cm = cm
        plotter.lines = lines
        plotter.patches = patches
        plotter.pp = pp
        plotter.show_figure = self.show_figure
        plotter.fill()
        fig = plotter.redraw()
        fio = io.BytesIO()
        fig.savefig(fio, format="png")
        fio.seek(0)
        if save_on_disk:
            tmp_file = tempfile.NamedTemporaryFile(
                suffix="%s.png" % plotter.__class__.__name__,
                delete=False)
            logging.info("Dumped figure to %s", tmp_file.name)
            tmp_file.write(fio.getvalue())
            fio.name = tmp_file.name
        return plotter, fio

    def compare_images(self, plotter, fio, tolerance=10):
        img1, img2 = [numpy.array(Image.open(fn)) for fn in (
            fio,
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "res/%s.png" % plotter.__class__.__name__))]
        diff = numpy.linalg.norm(img1 - img2)
        print("Difference:", diff)
        try:
            self.assertLess(diff, tolerance)
        except Exception as e:
            print(fio.name)
            raise from_none(e)
        finally:
            fio.close()
        if hasattr(fio, "name"):
            os.remove(fio.name)

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
        mp.reversed_labels_mapping = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
        self.compare_images(*self.run_plotter(mp))

    def testAccumulatingPlotter(self):
        ap = AccumulatingPlotter(self, name="Lines", clear_plot=True)
        ap.input = numpy.arange(1, 20, 0.1)
        ap.input_field = 0
        ap.label = "label name"
        ap, fio = self.run_plotter(ap)
        fig = None
        for i in range(11):
            ap.input_field = i + 1
            ap.fill()
            fig = ap.redraw()
        fig.savefig(fio, format="png")
        with open(fio.name, 'wb') as fout:
            fout.write(fio.getvalue())
        fio.seek(0)
        self.compare_images(ap, fio)

    def testImagePlotter(self):
        img = ImagePlotter(self, name="Image")
        img.inputs.append([numpy.zeros((100, 100))])
        img.inputs.append([numpy.ones((100, 100)) * 255])
        img.inputs[-1][0][0, 0] = 0
        img.input_fields.extend([0, 0])
        matplotlib.pyplot.switch_backend('agg')
        self.compare_images(*self.run_plotter(img))
        matplotlib.pyplot.switch_backend('cairo')

    def testImmediatePlotter(self):
        ip = ImmediatePlotter(self, name="Plot")
        ip.inputs.append([numpy.zeros(20) + 0.5])
        ip.inputs.append([numpy.arange(0, 1, 0.05)])
        ip.input_fields.extend([0, 0])
        self.compare_images(*self.run_plotter(ip))

    def testHistogram(self):
        h = Histogram(self, name="Histogram Test")
        h.gl_min = 0.0
        h.gl_max = 1.0
        h.x = numpy.arange(0, 1.1, 0.1)
        h.y = numpy.zeros(11)
        for i in numpy.arange(-1, 1.2, 0.2):
            h.y[int(numpy.round((i + 1) / 0.2))] = i * i
        self.compare_images(*self.run_plotter(h), tolerance=300)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
