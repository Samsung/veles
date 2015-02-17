"""
Created on Mar 7, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import time
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
from veles.memory import Vector
from veles.iplotter import IPlotter  # pylint: disable=W0611
from veles.graphics_server import GraphicsServer
from veles.units import Unit, IUnit


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
        self._last_run_ = 0
        self._server_ = None

    def __getstate__(self):
        state = super(Plotter, self).__getstate__()
        if self.stripped_pickle:
            for an, av in state.items():
                if isinstance(av, Vector):
                    state[an] = av.mem
        return state

    @property
    def last_run_time(self):
        return self._last_run_

    @last_run_time.setter
    def last_run_time(self, value):
        self._last_run_ = value

    @property
    def graphics_server(self):
        return self._server_

    @graphics_server.setter
    def graphics_server(self, value):
        if value is not None and not isinstance(value, GraphicsServer):
            raise TypeError("value must be of type veles.graphics_server."
                            "GraphicsServer (%s was specified)" % type(value))
        self._server_ = value

    def initialize(self, **kwargs):
        self._last_run_ = 0
        self._server_ = kwargs.get("graphics_server", self.graphics_server)

    def run(self):
        if self.workflow.plotters_are_enabled and \
           (time.time() - self._last_run_) > self.redraw_threshold:
            assert self.graphics_server is not None
            self._last_run_ = time.time()
            self.stripped_pickle = True
            self.graphics_server.enqueue(self)
            self.stripped_pickle = False

    def generate_data_for_master(self):
        return True

    def apply_data_from_slave(self, data, slave):
        if not self.gate_block and not self.gate_skip:
            self.run()
