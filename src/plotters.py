"""
Created on May 17, 2013

@author: Kumok Akim <a.kumok@samsung.com>
"""
import matplotlib
import matplotlib.pyplot as pp
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm
import pylab
import units
import tkinter
from PyQt4 import QtGui, QtCore
import queue
import threading
import thread_pool
import numpy
import logging
import config
import time


pp.ion()


init_lock = threading.Lock()


def norm_image(a):
    """Normalizes numpy array for passing to imshow.
    """
    a = a.copy()
    a -= a.min()
    m = a.max()
    if m:
        m /= 255.0
        a /= m
    else:
        a[:] = 127.5
    return a.astype(numpy.uint8)


class Graphics:
    """ Class handling all interaction with main graphics window
        NOTE: This class should be created ONLY within one thread
        (preferably main)

    Attributes:
        _instance: instance of MainGraphicsData class. Used for implementing
            Singleton pattern for this class.
        root: TKinter graphics root.
        event_queue: Queue of all pending changes created by other threads.
        run_lock: Lock to determine whether graphics window is running
        registered_plotters: List of registered plotters
        is_initialized: whether this class was already initialized.
    """

    _instance = None
    root = None
    event_queue = None
    run_lock = None
    registered_plotters = None
    is_initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Graphics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        global init_lock
        init_lock.acquire()
        if self.is_initialized:
            init_lock.release()
            return
        self.run_lock = threading.Lock()
        self.run_lock.acquire()
        self.is_initialized = True
        self.exiting = False
        self.event_queue = queue.Queue()
        self.initialize_lock = threading.Lock()
        self.registered_plotters = {}
        thread_pool.pool.request(self.run)
        init_lock.release()

    def run(self):
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        if pp.get_backend() == "TkAgg":
            self.root = tkinter.Tk()
            self.root.withdraw()
            self.root.after(100, self.update)
            tkinter.mainloop()  # Wait for user to close the window
        elif pp.get_backend() == "Qt4Agg":
            self.root = QtGui.QApplication([])
            self.timer = QtCore.QTimer(self.root)
            self.timer.timeout.connect(self.update)
            self.timer.start(100)
            self.root.exec_()
        elif pp.get_backend() == "WebAgg":
            self.exiting = False
            self.showed = False
            matplotlib.rcParams['webagg.port'] = 8888
            matplotlib.rcParams['webagg.open_in_browser'] = 'False'
            while not self.exiting:
                self.update()
                time.sleep(0.1)
        self.run_lock.release()

    def process_event(self, plotter):
        """Processes scheduled redraw event
        """
        plotter.redraw()

    def update(self):
        """Processes all events scheduled for plotting
        """
        try:
            while True:
                plotter = self.event_queue.get_nowait()
                self.process_event(plotter)
                if self.__dict__.get("showed") == False:
                    self.showed = True
                    threading.Thread(target=pp.show).start()
        except queue.Empty:
            pass
        if pp.get_backend() == "TkAgg":
            if not self.exiting:
                self.root.after(100, self.update)
            else:
                self.root.destroy()
        if pp.get_backend() == "Qt4Agg" and self.exiting:
            self.timer.stop()
            self.root.quit()

    def wait_finish(self):
        """Waits for user to close the window.
        """
        self.exiting = True
        if self.run_lock:
            self.run_lock.acquire()
            self.run_lock.release()
        logging.info("Done")


class Plotter(units.Unit):
    """Base class for all plotters

    Attributes:
        lock_: lock.
    """
    def __init__(self, device=None):
        super(Plotter, self).__init__()
        self.lock_ = threading.Lock()
        self.lock_.acquire()

    def redraw(self):
        """ Do the actual drawing here
        """
        self.lock_.release()

    def run(self):
        Graphics().event_queue.put(self, block=True)
        self.lock_.acquire()


class SimplePlotter(Plotter):
    """ Plotter for given values

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    Attributes:
        values: history of all parameter values given to plotter.
        input: connector to take values from.
        input_field: name of field in input we want to plot.
        figure_label: label of figure used for drawing. If two ploters share
            the same figure_label, their plots will appear together.
        plot_style: Style of lines used for plotting. See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
            for reference.
    """
    def __init__(self, figure_label="num errors",
                 plot_style="k-",
                 clear_plot=False,
                 bounds=None):
        super(SimplePlotter, self).__init__()
        self.values = []
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.plot_style = plot_style
        self.input_offs = 0
        self.clear_plot = clear_plot
        self.bounds = bounds

    def redraw(self):
        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        if self.clear_plot:
            figure.clf()
        axes = figure.add_subplot(111)  # Main axes
        if self.clear_plot:
            axes.cla()
        if self.bounds != None:
            axes.set_ylim(self.bounds[0], self.bounds[1])
        axes.plot(self.values, self.plot_style)
        figure.show()
        super(SimplePlotter, self).redraw()

    def run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = float(self.input[self.input_field])
        else:
            value = float(self.input.__dict__[self.input_field])
        if type(value) == numpy.ndarray:
            value = float(value[self.input_offs])
        self.values.append(value)
        super(SimplePlotter, self).run()


class MatrixPlotter(Plotter):
    """Plotter for drawing matrixes as table.

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Matrix"):
        super(MatrixPlotter, self).__init__()
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()
        num_rows = len(value) + 2
        num_columns = len(value[0]) + 2

        main_axes = figure.add_axes([0, 0, 1, 1])
        main_axes.cla()

        # First cell color
        rc = patches.Rectangle(
            (0, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First row last cell color
        rc = patches.Rectangle(
            ((num_columns - 1) / num_columns, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First column last cell color
        rc = patches.Rectangle(
            (0, 0),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # Last cell color
        rc = patches.Rectangle(
            ((num_columns - 1) / num_columns, 0),
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
                        v = int(numpy.clip(
                            numpy.round((1.0 - n_elem / max_vle) * 255.0),
                            0, 255))
                        color = "#FF%02X%02X" % (v, v)
                    else:
                        color = 'green'
                rc = patches.Rectangle(
                    (column / num_columns, (num_rows - row - 1) / num_rows),
                    1.0 / num_rows, 1.0 / num_columns, color=color)
                main_axes.add_patch(rc)

        for row in range(num_rows):
            y = row / num_rows
            main_axes.add_line(lines.Line2D([0, 1], [y, y]))
        for column in range(num_columns):
            x = column / num_columns
            main_axes.add_line(lines.Line2D([x, x], [0, 1]))

        # First cell
        column = 0
        row = 0
        pp.figtext(label="0",
            s="target",
            x=(column + 0.9) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="right")
        pp.figtext(label="0",
            s="value",
            x=(column + 0.1) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="left")

        # Headers in first row
        row = 0
        for column in range(1, num_columns - 1):
            pp.figtext(label=("C%d" % (column - 1)),
                            s=(column - 1),
                            x=(column + 0.5) / num_columns,
                            y=(num_rows - row - 0.5) / num_rows,
                            verticalalignment="center",
                            horizontalalignment="center")
        # Headers in first column
        column = 0
        for row in range(1, num_rows - 1):
            pp.figtext(label=("R%d" % (row - 1)),
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
                pp.figtext(
                    label=label,
                    s=n_elem,
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.33) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
                pp.figtext(
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
        pp.figtext(
            label=label,
            s=sum_ok,
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        pp.figtext(
            label=label,
            s=("%.2f%%" % (pt_total)),
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        figure.show()
        super(MatrixPlotter, self).redraw()


class Weights2D(Plotter):
    """Plotter for drawing weights as 2D.

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Weights", limit=64):
        super(Weights2D, self).__init__()
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.get_shape_from = None
        self.limit = limit
        self.transposed = False

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        if type(value) != numpy.ndarray or len(value.shape) != 2:
            return

        if self.transposed:
            value = value.transpose()

        if value.shape[0] > self.limit:
            value = value[:self.limit]

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        color = False
        if self.get_shape_from == None:
            sx = int(numpy.round(numpy.sqrt(value.shape[1])))
            sy = int(value.shape[1]) // sx
        elif type(self.get_shape_from) == list:
            sx = self.get_shape_from[0][1]
            sy = self.get_shape_from[0][0]
        elif "v" in self.get_shape_from.__dict__:
            if len(self.get_shape_from.v.shape) == 3:
                sx = self.get_shape_from.v.shape[2]
                sy = self.get_shape_from.v.shape[1]
            else:
                sx = self.get_shape_from.v.shape[3]
                sy = self.get_shape_from.v.shape[2]
                color = True
        else:
            sx = self.get_shape_from.shape[1]
            sy = self.get_shape_from.shape[0]

        if color:
            sz = sx * sy * 3
        else:
            sz = sx * sy

        n_cols = int(numpy.round(numpy.sqrt(value.shape[0])))
        n_rows = int(numpy.ceil(value.shape[0] / n_cols))

        i = 0
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                ax = figure.add_subplot(n_rows, n_cols, i + 1)
                ax.cla()
                ax.axis('off')
                v = value[i].ravel()[:sz]
                if color:
                    w = numpy.zeros([sy, sx, 3], dtype=v.dtype)
                    w[:, :, 0:1] = v.reshape(3, sy, sx)[0:1, :, :].reshape(sy,
                                                sx, 1)[:, :, 0:1]
                    w[:, :, 1:2] = v.reshape(3, sy, sx)[1:2, :, :].reshape(sy,
                                                sx, 1)[:, :, 0:1]
                    w[:, :, 2:3] = v.reshape(3, sy, sx)[2:3, :, :].reshape(sy,
                                                sx, 1)[:, :, 0:1]
                    ax.imshow(norm_image(w), interpolation="nearest")
                else:
                    ax.imshow(norm_image(v.reshape(sy, sx)),
                              interpolation="nearest", cmap=cm.gray)
                i += 1
                if i >= value.shape[0]:
                    break
            if i >= value.shape[0]:
                break

        figure.show()

        super(Weights2D, self).redraw()


class Image(Plotter):
    """Plotter for drawing N images.

    Should be assigned before initialize():
        inputs
        inputs_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image"):
        super(Image, self).__init__()
        self.inputs = []
        self.input_fields = []
        self.figure_label = figure_label

    def draw_image(self, ax, value):
        if type(value) != numpy.ndarray:
            ax.axis('off')
            ax.text(0.5, 0.5, "Unsupported type\n%s" % (
                str(type(value))), ha='center', va='center')
            return
        w = None
        color = False
        l = len(value.shape)
        if l == 2:
            sy = value.shape[0]
            sx = value.shape[1]
        elif l == 3:
            if value.shape[0] == 3:
                sy = value.shape[1]
                sx = value.shape[2]
                w = numpy.zeros([sy, sx, 3], dtype=value.dtype)
                w[:, :, 0:1] = value.reshape(3, sy, sx)[0:1, :, :].reshape(sy,
                                                    sx, 1)[:, :, 0:1]
                w[:, :, 1:2] = value.reshape(3, sy, sx)[1:2, :, :].reshape(sy,
                                                    sx, 1)[:, :, 0:1]
                w[:, :, 2:3] = value.reshape(3, sy, sx)[2:3, :, :].reshape(sy,
                                                    sx, 1)[:, :, 0:1]
                color = True
            elif value.shape[2] == 3:
                sy = value.shape[0]
                sx = value.shape[1]
                color = True
            else:
                sy = int(numpy.round(numpy.sqrt(value.size)))
                sx = int(numpy.round(value.size / sy))
                value = value.reshape(sy, sx)
        else:
            sy = int(numpy.round(numpy.sqrt(value.size)))
            sx = int(numpy.round(value.size / sy))
            value = value.reshape(sy, sx)

        if w == None:
            w = value.copy()

        if color:
            ax.imshow(norm_image(w), interpolation="nearest")
        else:
            ax.imshow(norm_image(w), interpolation="nearest", cmap=cm.gray)

    def redraw(self):
        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        for i, input_field in enumerate(self.input_fields):
            value = None
            if type(input_field) == int:
                if input_field >= 0 and input_field < len(self.inputs[i]):
                    value = self.inputs[i][input_field]
            else:
                value = self.inputs[i].__dict__[input_field]
            ax = figure.add_subplot(len(self.input_fields), 1, i + 1)
            ax.cla
            self.draw_image(ax, value)

        figure.show()
        super(Image, self).redraw()


class MSEHistogram(Plotter):
    """Plotter for drawing histogram.

    Should be assigned before initialize():
        mse

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Histogram", n_bars=35):
        super(MSEHistogram, self).__init__()
        self.figure_label = figure_label
        self.val_mse = None
        self.mse_min = None
        self.mse_max = None
        self.n_bars = n_bars
        self.mse = None  # formats.Vector()

    def initialize(self):
        self.val_mse = numpy.zeros(self.n_bars,
                                   dtype=config.dtypes[config.dtype])

    def redraw(self):
        fig = pp.figure(self.figure_label)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        # fig.patch.set_alpha(0.45)

        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.patch.set_facecolor('#ffe6ca')
        # ax.patch.set_alpha(0.45)

        ymin = self.val_min
        ymax = (self.val_max) * 1.3
        xmin = self.mse_min
        xmax = self.mse_max
        width = ((xmax - xmin) / self.n_bars) * 0.8
        t0 = 0.65 * ymax
        l1 = width * 0.5

        if self.n_bars < 11:
            l3 = 20
            koef = 0.5 * ymax
            l2 = 0.235 * ymax

        if self.n_bars < 31 and self.n_bars > 10:
            l3 = 25 - (0.5) * self.n_bars
            koef = 0.635 * ymax - 0.0135 * self.n_bars * ymax
            l2 = 0.2975 * ymax - 0.00625 * self.n_bars * ymax

        if self.n_bars < 41 and self.n_bars > 30:
            l3 = 16 - (0.2) * self.n_bars
            koef = 0.32 * ymax - 0.003 * self.n_bars * ymax
            l2 = 0.17 * ymax - 0.002 * self.n_bars * ymax

        if self.n_bars < 51 and self.n_bars > 40:
            l3 = 8
            koef = 0.32 * ymax - 0.003 * self.n_bars * ymax
            l2 = 0.17 * ymax - 0.002 * self.n_bars * ymax

        if self.n_bars > 51:
            l3 = 8
            koef = 0.17 * ymax
            l2 = 0.07 * ymax

        N = numpy.linspace(self.mse_min, self.mse_max, num=self.n_bars,
                           endpoint=True)
        ax.bar(N, self.val_mse, color='#ffa0ef', width=width,
               edgecolor='lavender')
        # , edgecolor='red')
        # D889B8
        # B96A9A
        ax.set_xlabel('Errors', fontsize=20)
        ax.set_ylabel('Input Data', fontsize=20)
        ax.set_title('Histogram', fontsize=25)
        ax.axis([xmin, xmax + ((xmax - xmin) / self.n_bars), ymin, ymax])
        ax.grid(True)
        leg = ax.legend((self.figure_label.replace("Histogram ", "")))
                        # 'upper center')
        frame = leg.get_frame()
        frame.set_facecolor('#E8D6BB')
        for t in leg.get_texts():
            t.set_fontsize(18)
        for l in leg.get_lines():
            l.set_linewidth(1.5)

        for x, y in zip(N, self.val_mse):
            if y > koef - l2:
                pylab.text(x + l1, y - l2, '%.2f' % y,
                           ha='center', va='bottom',
                           fontsize=l3, rotation=90)
            else:
                pylab.text(x + l1, t0, '%.2f' % y,
                           ha='center', va='bottom',
                           fontsize=l3, rotation=90)

        fig.show()
        super(MSEHistogram, self).redraw()

    def run(self):
        mx = self.mse.v.max()
        mi = self.mse.v.min()
        self.mse_max = mx
        self.mse_min = mi
        d = mx - mi
        if not d:
            return
        d = (self.n_bars - 1) / d
        self.val_mse[:] = 0
        for mse in self.mse.v:
            i_bar = int(numpy.floor((mse - mi) * d))
            self.val_mse[i_bar] += 1

        self.val_max = self.val_mse.max()
        self.val_min = self.val_mse.min()

        super(MSEHistogram, self).run()
