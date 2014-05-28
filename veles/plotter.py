"""
Created on Mar 7, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import time
from zope.interface import Interface, Attribute, implementer

from veles.distributable import TriviallyDistributable
from veles.graphics_server import GraphicsServer
from veles.units import Unit, IUnit


class IPlotter(Interface):
    """Plots stuff in GraphicsClient environment.
    """

    matplotlib = Attribute("""matplotlib module reference""")
    cm = Attribute("""matplotlib.cm (colormap) module reference""")
    lines = Attribute("""matplotlib.lines module reference""")
    patches = Attribute("""matplotlib.patches module reference""")
    pp = Attribute("""matplotlib.pyplot module reference""")

    def redraw():
        """Updates the plot using the changed object's state.
        Should be implemented by the class providing this interface.
        """

    def show_figure(figure):
        """figure.show() non-blocking wrapper. Added automatically, must not be
        implemented.
        """


@implementer(IUnit)
class Plotter(Unit, TriviallyDistributable):
    """Base class for all plotters.
    """
    server_shutdown_registered = False

    def __init__(self, workflow, **kwargs):
        view_group = kwargs.get("view_group", "PLOTTER")
        kwargs["view_group"] = view_group
        super(Plotter, self).__init__(workflow, **kwargs)
        self.redraw_threshold = 0.5

    def initialize(self, **kwargs):
        self.last_run_ = time.time()

    def run(self):
        if self.workflow.plotters_are_enabled and \
           (time.time() - self.last_run_) > self.redraw_threshold:
            self.last_run_ = time.time()
            self.stripped_pickle = True
            GraphicsServer().enqueue(self)
            self.stripped_pickle = False

    def generate_data_for_master(self):
        return True

    def apply_data_from_slave(self, data, slave):
        if not self.gate_block and not self.gate_skip:
            self.run()
