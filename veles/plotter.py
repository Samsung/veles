"""
Created on Mar 7, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import config
from graphics_server import GraphicsServer
import time
from units import Unit


class Plotter(Unit):
    """Base class for all plotters.
    """
    server_shutdown_registered = False

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name")
        view_group = kwargs.get("view_group", "PLOTTER")
        kwargs["name"] = name
        kwargs["view_group"] = view_group
        super(Plotter, self).__init__(workflow, **kwargs)
        self.stripped_pickle = False
        self.last_run = time.time()
        self.redraw_threshold = 0.1

    def redraw(self):
        """ Do the actual drawing here
        """
        pass

    def __getstate__(self):
        state = super(Plotter, self).__getstate__()
        if self.stripped_pickle:
            state["links_from"] = {}
            state["links_to"] = {}
            state["_workflow"] = None
        return state

    def run(self):
        if self.workflow.plotters_are_enabled and \
           (time.time() - self.last_run) > self.redraw_threshold:
            self.last_run = time.time()
            self.stripped_pickle = True
            GraphicsServer().enqueue(self)
            self.stripped_pickle = False
            if self.should_unlock_pipeline:
                self.workflow.unlock_pipeline()

    def generate_data_for_master(self):
        return True

    def apply_data_from_slave(self, data, slave=None):
        if ((((not Unit.callvle(self.gate_block[0])) and
              (not Unit.callvle(self.gate_block_not[0]))) or
             (Unit.callvle(self.gate_block[0]) and
              Unit.callvle(self.gate_block_not[0]))) and
            (((not Unit.callvle(self.gate_skip[0])) and
              (not Unit.callvle(self.gate_skip_not[0]))) or
             (Unit.callvle(self.gate_skip[0]) and
              Unit.callvle(self.gate_skip_not[0])))):
            self.run()
