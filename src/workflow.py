"""
Created on Aug 6, 2013

Base class for workflows.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units


class Workflow(units.Unit):
    """Base class for workflows.

    Attributes:
        start_point: start point.
        end_point: end point.
    """
    def __init__(self):
        super(Workflow, self).__init__()
        self.start_point = units.Unit()
        self.end_point = units.EndPoint()

    def run(self):
        """Do the job here.

        In the child class:
            call the parent method at the end.
        """
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


class NNWorkflow(units.OpenCLUnit, Workflow):
    """Base class for neural network workflows.

    Attributes:
        rpt: repeater.
        loader: loader unit.
        forward: list of the forward units.
        ev: evaluator unit.
        decision: decision unit.
        gd: list of the gradient descent units.
    """
    def __init__(self, device=None):
        super(NNWorkflow, self).__init__(device=device)
        self.rpt = units.Repeater()
        self.loader = None
        self.forward = []
        self.ev = None
        self.decision = None
        self.gd = []

    def initialize(self, device=None):
        super(NNWorkflow, self).initialize()
        if device != None:
            self.device = device
        for obj in self.forward:
            if obj != None:
                obj.device = self.device
        if self.ev != None:
            self.ev.device = self.device
        for obj in self.gd:
            if obj != None:
                obj.device = self.device
