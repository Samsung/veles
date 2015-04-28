# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 17, 2013

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


from __future__ import division
from collections import defaultdict
from datetime import datetime
import numpy
import time
from zope.interface import implementer

import veles.error as error
import veles.memory as formats
from veles.compat import from_none
from veles.distributable import IDistributable
from veles.mutable import Bool
from veles.plotter import Plotter, IPlotter
from veles.units import nothing


@implementer(IPlotter)
class AccumulatingPlotter(Plotter):
    """Accumulates supplied values and draws the plot of the last "last"
    points, as well as the whole picture in miniature. Optionally, approximates
    the series with a polynomial of power "fit_poly_power" in terms of least
    squares.

    Must be assigned before initialize():
        input
        input_field ^

    ^ If input_field is not assigned, input is treated as a single floating
      point value.

    Updates after run():

    Creates within initialize():

    Attributes:
        values: history of all parameter values given to plotter.
        input: connector to take values from.
        input_field: name of field in input we want to plot.
        name(): label of figure used for drawing. If two plotters share
            the same name(), their plots will appear together.
        plot_style: Style of lines used for plotting. See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
            for reference.
        clear_plot: will clear plot at the beginning of redraw().
        redraw_plot: will redraw plot at the end of redraw().
        ylim: bounds of plot y-axis.
        last: the number of ending points to show. If set to 0, always plot
        all the accumulated points.
        minimap_size: draw the miniature plot of the whole series in the upper
        right corner of this size. If set to 0, no minimap is drawn.
        fit_poly_power: the approximation polynomial's power. If set to 0,
        do no approximation.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "AccumulatingPlotter")
        super(AccumulatingPlotter, self).__init__(workflow, **kwargs)
        self.plot_style = kwargs.get("plot_style", "k-")
        self.clear_plot = kwargs.get("clear_plot", False)
        self.redraw_plot = kwargs.get("redraw_plot", False)
        self.ylim = kwargs.get("ylim")
        self.last = kwargs.get("last", 11)
        self.fit_poly_power = kwargs.get("fit_poly_power", 2)
        self.minimap_size = kwargs.get("minimap", 0.25)
        self.line_width = kwargs.get("line_width", 2.0)
        self.values = []
        self.input_offset = 0
        self.pp = None
        self.show_figure = nothing
        self.input_field = None
        self.demand("input")

    def redraw(self):
        self.pp.ioff()
        figure = self.pp.figure(self.name)
        if self.clear_plot:
            figure.clf()
        axes = figure.add_subplot(111)  # Main axes
        axes.grid(True)
        if self.ylim is not None:
            axes.set_ylim(self.ylim[0], self.ylim[1])
        if self.last == 0:
            axes.plot(self.values, self.plot_style)
        else:
            values = self.values[-self.last:]
            if len(self.values) > self.last:
                begindex = len(self.values) - self.last
                values_range = numpy.arange(len(values)) + begindex
                axes.xaxis.set_ticks(range(begindex, len(self.values)))
                if self.fit_poly_power == 0:
                    axes.plot(values_range, values, self.plot_style[:-1] + '-',
                              linewidth=self.line_width, marker='o',
                              markersize=self.line_width * 4)
                else:
                    interval = numpy.linspace(0, len(values) - 1, 100)
                    smooth_vals = numpy.poly1d(numpy.polyfit(
                        range(len(values)), values,
                        self.fit_poly_power))(interval)
                    axes.plot(interval + begindex, smooth_vals,
                              self.plot_style, linewidth=self.line_width)
                    axes.plot(values_range, values, self.plot_style[:-1] + 'o',
                              linewidth=self.line_width * 4)
                if self.minimap_size > 0:
                    minimap = figure.add_axes((1 - self.minimap_size,
                                               1 - self.minimap_size,
                                               self.minimap_size,
                                               self.minimap_size),
                                              alpha=0.75)
                    minimap.xaxis.set_visible(False)
                    minimap.yaxis.set_visible(False)
                    minimap.plot(self.values, self.plot_style,
                                 linewidth=self.line_width)
            else:
                axes.xaxis.set_ticks(range(len(values)))
                axes.plot(values, self.plot_style[:-1] + '-',
                          linewidth=self.line_width, marker='o',
                          markersize=self.line_width * 4)
        self.pp.ion()
        self.show_figure(figure)
        if self.redraw_plot:
            figure.canvas.draw()
        return figure

    def fill(self):
        if self.input_field is None:
            try:
                value = float(self.input)
            except TypeError:
                raise from_none(TypeError("input has a wrong type %s - must be"
                                          " float" % self.input.__class__))
        elif isinstance(self.input_field, int):
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]
        if type(value) == numpy.ndarray:
            value = value[self.input_offset]
        self.values.append(float(value))


@implementer(IPlotter)
class MatrixPlotter(Plotter):
    """Plotter for drawing matrixes as table.

    Must be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "MatrixPlotter")
        kwargs["name"] = name
        super(MatrixPlotter, self).__init__(workflow, **kwargs)
        self.input = None  # Connector
        self.input_field = None
        self.pp = None
        self.patches = None
        self.lines = None
        self.show_figure = nothing
        self.demand("input", "input_field", "reversed_labels_mapping")

    def redraw(self):
        self.pp.ioff()
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        figure = self.pp.figure(self.name)
        figure.clf()
        main_axes = figure.add_axes([0, 0, 1, 1])
        main_axes.cla()

        num_rows = len(value) + 2
        num_columns = len(value[0]) + 2
        # First cell color
        rc = self.patches.Rectangle(
            (0, (num_rows - 1.0) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First row last cell color
        rc = self.patches.Rectangle(
            ((num_columns - 1.0) / num_columns, (num_rows - 1.0) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First column last cell color
        rc = self.patches.Rectangle(
            (0, 0),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # Last cell color
        rc = self.patches.Rectangle(
            ((num_columns - 1.0) / num_columns, 0),
            1.0 / num_rows, 1.0 / num_columns, color='silver')
        main_axes.add_patch(rc)
        # Data cells colors
        sum_total = value.sum()
        sum_ok = 0
        max_vle = 0
        for row in range(0, num_rows - 2):
            for column in range(0, num_columns - 2):
                if row != column:
                    max_vle = max(max_vle, value[row, column])
                else:
                    sum_ok += value[row, column]
        # sum_by_row = value.sum(axis=0)
        for row in range(1, num_rows - 1):
            for column in range(1, num_columns - 1):
                n_elem = value[row - 1, column - 1]
                color = 'white'
                if row == column:
                    if n_elem > 0:
                        color = 'cyan'
                else:
                    if n_elem > 0:
                        v = int(numpy.clip(numpy.round(
                            (1.0 - (n_elem + 0.0) / max_vle) * 255.0),
                            0, 255))
                        color = "#FF%02X%02X" % (v, v)
                    else:
                        color = 'green'
                rc = self.patches.Rectangle(
                    ((column + 0.0) / num_columns,
                     (num_rows - row - 1.0) / num_rows),
                    1.0 / num_rows, 1.0 / num_columns, color=color)
                main_axes.add_patch(rc)

        for row in range(num_rows):
            y = (row + 0.0) / num_rows
            main_axes.add_line(self.lines.Line2D([0, 1], [y, y]))
        for column in range(num_columns):
            x = (column + 0.0) / num_columns
            main_axes.add_line(self.lines.Line2D([x, x], [0, 1]))

        # First cell
        column = 0
        row = 0
        figure.text(
            label="0",
            s="target",
            x=(column + 0.9) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="right")
        figure.text(
            label="0",
            s="value",
            x=(column + 0.1) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="left")
        # Headers in first row
        row = 0
        for column in range(1, num_columns - 1):
            label = self.reversed_labels_mapping[column - 1]
            figure.text(label="C%s" % str(label),
                        s=(column - 1),
                        x=(column + 0.5) / num_columns,
                        y=(num_rows - row - 0.5) / num_rows,
                        verticalalignment="center",
                        horizontalalignment="center")
        # Headers in first column
        column = 0
        for row in range(1, num_rows - 1):
            label = self.reversed_labels_mapping[row - 1]
            figure.text(label=("R%s" % str(label)),
                        s=(row - 1),
                        x=(column + 0.5) / num_columns,
                        y=(num_rows - row - 0.5) / num_rows,
                        verticalalignment="center",
                        horizontalalignment="center")
        # Data
        for row in range(1, num_rows - 1):
            for column in range(1, num_columns - 1):
                n_elem = value[row - 1, column - 1]
                n = sum_total
                pt_total = 100.0 * n_elem / n if n else 0
                label = "%d as %d" % (column - 1, row - 1)
                figure.text(
                    label=label,
                    s=n_elem,
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.33) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
                figure.text(
                    label=label,
                    s=("%.2f%%" % (pt_total)),
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.66) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
        # Last cell
        n = sum_total
        pt_total = 100.0 * sum_ok / n if n else 0
        label = "Totals"
        row = num_rows - 1
        column = num_columns - 1
        figure.text(
            label=label,
            s=sum_ok,
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        figure.text(
            label=label,
            s=("%.2f%%" % (pt_total)),
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        self.pp.ion()
        self.show_figure(figure)
        figure.canvas.draw()
        return figure


@implementer(IPlotter)
class ImagePlotter(Plotter):
    """Plotter for drawing N images.

    Must be assigned before initialize():
        inputs
        input_fields

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "ImagePlotter")
        super(ImagePlotter, self).__init__(workflow, **kwargs)
        self.yuv = Bool(kwargs.get("yuv", False))
        self.cm = None
        self.pp = None
        self.show_figure = nothing
        self.demand("inputs", "input_fields")
        self.inputs = []
        self.input_fields = []
        self._pics_to_draw = []
        self.redraw_threshold = 1.5

    def __getstate__(self):
        state = super(ImagePlotter, self).__getstate__()
        if self.stripped_pickle:
            pics_to_draw = []
            for i, input_field in enumerate(self.input_fields):
                value = None
                if type(input_field) == int:
                    if input_field >= 0 and input_field < len(self.inputs[i]):
                        value = self.inputs[i][input_field]
                else:
                    value = self.inputs[i].__dict__[input_field]
                    if isinstance(self.inputs[i], formats.Vector):
                        value = value[0]
                pics_to_draw.append(self._prepare_image(value)
                                    if type(value) == numpy.ndarray
                                    else str(value))

            state["inputs"] = None
            state["input_fields"] = None
            state["_pics_to_draw"] = pics_to_draw
        return state

    def _prepare_image(self, value):
        l = len(value.shape)
        if l == 2:
            sy = value.shape[0]
            sx = value.shape[1]
        elif l == 3:
            if value.shape[0] == 3:
                sy = value.shape[1]
                sx = value.shape[2]
                w = numpy.zeros([sy, sx, 3], dtype=value.dtype)
                w[:, :, 0:1] = value.reshape(
                    3, sy, sx)[0:1, :, :].reshape(sy, sx, 1)[:, :, 0:1]
                w[:, :, 1:2] = value.reshape(
                    3, sy, sx)[1:2, :, :].reshape(sy, sx, 1)[:, :, 0:1]
                w[:, :, 2:3] = value.reshape(
                    3, sy, sx)[2:3, :, :].reshape(sy, sx, 1)[:, :, 0:1]
                value = w
            elif value.shape[2] == 3:
                sy = value.shape[0]
                sx = value.shape[1]
            else:
                sy = int(numpy.round(numpy.sqrt(value.size)))
                sx = int(numpy.round(value.size / sy))
                value = value.reshape(sy, sx)
        else:
            sy = int(numpy.round(numpy.sqrt(value.size)))
            sx = int(numpy.round(value.size / sy))
            value = value.reshape(sy, sx)

        # Normalize value and convert to uint8
        value = value.astype(numpy.float32).copy()
        value -= value.min()
        mx = value.max()
        if mx:
            value *= 255.0 / mx
        value = numpy.round(value).astype(numpy.uint8)

        if not self.yuv:
            return value

        import cv2
        return cv2.cvtColor(value, cv2.COLOR_YUV2RGB)

    def redraw(self):
        figure = self.pp.figure(self.name)
        figure.clf()

        for i, pic in enumerate(self._pics_to_draw):
            ax = figure.add_subplot(len(self._pics_to_draw), 1, i + 1)
            ax.cla()
            if type(pic) != numpy.ndarray:
                ax.axis('off')
                ax.text(0.5, 0.5, str(pic), ha='center', va='center')
                continue
            if len(pic.shape) == 3:
                ax.imshow(pic, interpolation="nearest")
            else:
                ax.imshow(pic, interpolation="nearest", cmap=self.cm.gray)

        self.show_figure(figure)
        figure.canvas.draw()
        return figure


@implementer(IPlotter)
class ImmediatePlotter(Plotter):
    """Plotter for drawing N plots together.

    Must be assigned before initialize():
        inputs
        input_fields
        input_styles

    Updates after run():

    Creates within initialize():

    Attributes:
        inputs: list of inputs.
        input_fields: list of fields for corresponding input.
        input_styles: pyplot line styles for corresponding input.
        ylim: bounds of the plot y-axis.
    """

    DEFAULT_STYLES = ["k-", "g-", "b-"]

    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name")
        super(ImmediatePlotter, self).__init__(workflow, **kwargs)
        self.inputs = []
        self.input_fields = []
        self.input_styles = []
        self.ylim = kwargs.get("ylim")
        self.pp = None
        self.show_figure = nothing

    def redraw(self):
        figure = self.pp.figure(self.name)
        figure.clf()
        ax = figure.add_subplot(111)
        ax.cla()
        if self.ylim is not None:
            ax.set_ylim(self.ylim[0], self.ylim[1])

        for i, input_field in enumerate(self.input_fields):
            value = None
            if type(input_field) == int:
                if input_field >= 0 and input_field < len(self.inputs[i]):
                    value = self.inputs[i][input_field]
            else:
                value = self.inputs[i].__dict__[input_field]

            ax.plot(value, self.input_styles[i] if i < len(self.input_styles)
                    else ImmediatePlotter.DEFAULT_STYLES[i])

        self.show_figure(figure)
        figure.canvas.draw()
        return figure


@implementer(IPlotter)
class Histogram(Plotter):
    """
    Plotter for drawing histogram.
    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Histogram")
        kwargs["name"] = name
        super(Histogram, self).__init__(workflow, **kwargs)
        self.gl_min = 0
        self.gl_max = 1
        self.demand("x", "y")
        self.pp = None
        self.show_figure = nothing

    def redraw(self):
        fig = self.pp.figure(self.name)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        # fig.patch.set_alpha(0.45)

        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.patch.set_facecolor('#ffe6ca')
        # ax.patch.set_alpha(0.45)

        if len(self.x) != len(self.y):
            raise error.BadFormatError(
                "Shape of X %s not equal shape of Y %s !" %
                (len(self.x), len(self.y)))
        ymax = numpy.max(self.y) * 1.3
        ymin = numpy.min(self.y)
        xmax = numpy.max(self.x)
        xmin = numpy.min(self.x)
        nbars = len(self.x)

        width = ((xmax - xmin) / nbars) * 0.8
        t0 = 0.65 * ymax
        l1 = width * 0.5

        if nbars < 11:
            l3 = 20
            koef = 0.5 * ymax
            l2 = 0.235 * ymax

        if nbars < 31 and nbars > 10:
            l3 = 25 - (0.5) * nbars
            koef = 0.635 * ymax - 0.0135 * nbars * ymax
            l2 = 0.2975 * ymax - 0.00625 * nbars * ymax

        if nbars < 41 and nbars > 30:
            l3 = 16 - (0.2) * nbars
            koef = 0.32 * ymax - 0.003 * nbars * ymax
            l2 = 0.17 * ymax - 0.002 * nbars * ymax

        if nbars < 51 and nbars > 40:
            l3 = 8
            koef = 0.32 * ymax - 0.003 * nbars * ymax
            l2 = 0.17 * ymax - 0.002 * nbars * ymax

        if nbars > 51:
            l3 = 8
            koef = 0.17 * ymax
            l2 = 0.07 * ymax

        width = ((xmax - xmin) / nbars) * 0.8
        N = numpy.linspace(xmin, xmax, num=nbars,
                           endpoint=True)
        ax.bar(N, self.y, color='#ffa0ef', width=width,
               edgecolor='lavender')
        # , edgecolor='red')
        # D889B8
        # B96A9A
        ax.set_xlabel("X min = %.6g, max = %.6g" %
                      (self.gl_min, self.gl_max), fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.axis([xmin, xmax + ((xmax - xmin) / nbars), ymin, ymax])
        ax.grid(True)
        leg = ax.legend("Y ")
                        # 'upper center')
        frame = leg.get_frame()
        frame.set_facecolor('#E8D6BB')
        for t in leg.get_texts():
            t.set_fontsize(18)
        for l in leg.get_lines():
            l.set_linewidth(1.5)

        for x, y in zip(N, self.y):
            if y > koef - l2 * 0.75:
                self.pp.text(x + l1, y - l2 * 0.75, '%.0f' % y, ha='center',
                             va='bottom', fontsize=l3, rotation=90)
            else:
                self.pp.text(x + l1, t0, '%.0f' % y, ha='center', va='bottom',
                             fontsize=l3, rotation=90)

        self.show_figure(fig)
        fig.canvas.draw()
        return fig


@implementer(IPlotter)
class MultiHistogram(Plotter):
    """Plotter for drawing weights as 2D.

    Must be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Histogram")
        limit = kwargs.get("limit", 64)
        n_bars = kwargs.get("n_bars", 25)
        hist_number = kwargs.get("hist_number", 16)
        kwargs["name"] = name
        kwargs["limit"] = limit
        kwargs["n_bars"] = n_bars
        kwargs["hist_number"] = hist_number
        super(MultiHistogram, self).__init__(workflow, **kwargs)
        self.limit = limit
        self.pp = None
        self.show_figure = nothing
        self.input = None  # formats.Vector()
        self.value = formats.Vector()
        self.n_bars = n_bars
        self.hist_number = hist_number

    def initialize(self, **kwargs):
        super(MultiHistogram, self).initialize(**kwargs)
        if self.hist_number > self.limit:
            self.hist_number = self.limit
        self.value.mem = numpy.zeros(
            [self.hist_number, self.n_bars], dtype=numpy.int64)

    def redraw(self):

        fig = self.pp.figure(self.name)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        # fig.patch.set_alpha(0.45)

        n_cols = int(numpy.round(numpy.sqrt(self.value.shape[0])))
        n_rows = int(numpy.ceil(self.value.shape[0] / n_cols))
        i = 0
        for _ in range(0, n_rows):
            for _ in range(0, n_cols):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.cla()
                # ax.axis('off')
                ax.patch.set_facecolor('#ffe6ca')
                # ax.set_xlabel("Input Data", fontsize=10)
                # ax.set_ylabel("Number", fontsize=10)
                ymin = self.value[i].min()
                ymax = self.value[i].max()
                xmin = self.input[i].min()
                xmax = self.input[i].max()
                ax.axis([xmin, xmax + ((xmax - xmin) / self.n_bars), ymin,
                         ymax])
                ax.grid(True)
                ax.set_title(self.name.replace("Histogram ", ""))
                nbars = self.n_bars
                width = ((xmax - xmin) / nbars) * 0.8
                X = numpy.linspace(xmin, xmax, num=nbars, endpoint=True)
                Y = self.value[i]
                if (n_rows > 5) or (n_cols > 5):
                    ax.bar(X, Y, color='#ffa0ef', width=width,
                           edgecolor='red')
                else:
                    ax.bar(X, Y, color='#ffa0ef', width=width,
                           edgecolor='lavender')
                if n_rows > 4:
                    ax.set_yticklabels([])
                if n_cols > 3:
                    ax.set_xticklabels([])
                i += 1
                if i >= self.value.shape[0]:
                    break
            if i >= self.value.shape[0]:
                break

        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def fill(self):
        for i in range(self.hist_number):
            self.value.map_write()
            self.input.map_read()
            mx = self.input.mem[i].max()
            mi = self.input.mem[i].min()
            d = mx - mi
            if not d:
                return
            d = (self.n_bars - 1) / d
            self.value[i] = 0
            for x in self.input.mem[i]:
                i_bar = int(numpy.floor((x - mi) * d))
                self.value[i, i_bar] += 1


@implementer(IPlotter)
class TableMaxMin(Plotter):
    """
    Plotter for drawing histogram.
    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Table")
        kwargs["name"] = name
        super(TableMaxMin, self).__init__(workflow, **kwargs)
        self.row_labels = ["max", "min"]
        self.col_labels = []
        self.y = []
        self.values = formats.Vector()
        self.pp = None
        self.show_figure = nothing

    def initialize(self, **kwargs):
        super(TableMaxMin, self).initialize(**kwargs)
        self.values.mem = numpy.zeros(
            [2, len(self.y)], dtype=numpy.float64)

    def redraw(self):
        fig = self.pp.figure(self.name)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.patch.set_facecolor('#ffe6ca')
        vals = []
        for row in self.values:
            vals.append(list("%.6f" % x for x in row))
        the_table = ax.table(
            cellText=vals,
            # colWidths=[0.1] * len(self.y),
            rowLabels=self.row_labels, colLabels=self.col_labels,
            loc='center right')
        the_table.set_fontsize(36)
        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def fill(self):
        if len(self.col_labels) != len(self.y):
            raise error.BadFormatError(
                "Shape of col_names %s not equal shape of Y %s !" %
                (len(self.col_labels), len(self.y)))
        for i, y in enumerate(self.y):
            if self.y[i] is not None:
                self.values.mem[0, i] = y.mem.max()
                self.values.mem[1, i] = y.mem.min()
            else:
                self.values[0, i] = "None"
                self.values[1, i] = "None"


@implementer(IPlotter, IDistributable)
class SlaveStats(Plotter):
    """
    Draws slave iteration timings.
    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Slave Iteration Timings")
        kwargs["name"] = name
        super(SlaveStats, self).__init__(workflow, **kwargs)
        self._slave_timings = defaultdict(list)
        self._last_update_times_ = {}
        self.period = kwargs.get('period', 100)
        self._slaves = {}
        self.update_interval = kwargs.get('update_interval', 1.0)

    def redraw(self):
        if len(self._slaves) == 0:
            return None
        fig = self.pp.figure(self.name)
        fig.clf()
        axes = fig.add_subplot(111)
        minmax = []
        for sid, timings in self._slave_timings.items():
            minmax.append((timings[max(0, len(timings) - self.period)][1],
                           sid, False))
            minmax.append((timings[-1][1], sid, True))
        minmax.sort(key=lambda p: p[0], reverse=True)
        sid_white_list = set()
        start_time = None
        for p in minmax:
            if not p[2]:
                start_time = p[0]
                break
            sid_white_list.add(p[1])
        for sid in sorted(sid_white_list):
            timings = self._slave_timings[sid]
            slave = self._slaves[sid]
            y, x = ([t[i] for t in timings[-self.period:]
                     if t[1] >= start_time] for i in (0, 1))
            axes.plot(x, y, label="%s (%s:%d)" % (sid, slave.host, slave.pid))
        for sid in sorted(set(self._slave_timings.keys()) - sid_white_list):
            slave = self._slaves[sid]
            axes.plot(None, label="%s (%s:%d)" % (sid, slave.host, slave.pid))
        axes.set_ylim(bottom=0)
        axes.legend()
        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def generate_data_for_master(self):
        return {}

    def generate_data_for_slave(self, slave):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            return
        now = time.time()
        delta = now - self._last_update_times_[slave.id] \
            if slave.id in self._last_update_times_ else None
        self._last_update_times_[slave.id] = now
        if delta is None:
            self._slaves[slave.id] = slave
            return
        self._slave_timings[slave.id].append((delta,
                                              datetime.fromtimestamp(now)))
        if len(self._slave_timings[slave.id]) > 2 * self.period:
            self._slave_timings[slave.id] = \
                self._slave_timings[slave.id][-self.period:]
        if now - self._last_run_ > self.update_interval:
            self.run()

    def drop_slave(self, slave):
        if slave.id in self._slaves:
            del self._slaves[slave.id]
        if slave.id in self._last_update_times_:
            del self._last_update_times_[slave.id]
        if slave.id in self._slave_timings:
            del self._slave_timings[slave.id]
