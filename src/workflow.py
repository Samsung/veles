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
        self.start_point.run_dependent()
        self.end_point.wait()
