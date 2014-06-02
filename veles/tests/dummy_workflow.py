"""
Created on Mar 13, 2014

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

    def add_ref(self, workflow):
        pass

    def on_workflow_finished(self):
        pass

    def stop(self):
        pass


class DummyWorkflow(Workflow):
    """
    Dummy standalone workflow for unit tests.
    """

    def __init__(self):
        """
        Passes DummyLauncher as workflow parameter value.
        """
        super(DummyWorkflow, self).__init__(DummyLauncher())
