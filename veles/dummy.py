"""
Created on Oct 31, 2014

Dummy units for tests and benchmarks.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.workflow import Workflow


class DummyLauncher(object):
    @property
    def is_slave(self):
        return False

    @property
    def is_master(self):
        return False

    @property
    def is_standalone(self):
        return True

    @property
    def log_id(self):
        return "DUMMY"

    def add_ref(self, workflow):
        self.workflow = workflow

    def on_workflow_finished(self):
        pass

    def stop(self):
        pass


class DummyWorkflow(Workflow):
    """
    Dummy standalone workflow for tests and benchmarks.
    """

    def __init__(self):
        """
        Passes DummyLauncher as workflow parameter value.
        """
        super(DummyWorkflow, self).__init__(DummyLauncher())
        self.end_point.link_from(self.start_point)
