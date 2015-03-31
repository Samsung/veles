"""
Created on Oct 31, 2014

Dummy units for tests and benchmarks.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
from zope.interface import implementer

from veles.units import IUnit, TrivialUnit
from veles.workflow import Workflow


class DummyLauncher(object):
    def __init__(self):
        self.stopped = False
        self.interactive = False

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

    def del_ref(self, unit):
        pass

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
        self.launcher = DummyLauncher()
        super(DummyWorkflow, self).__init__(self.launcher)
        self.end_point.link_from(self.start_point)


@implementer(IUnit)
class DummyUnit(TrivialUnit):
    """
    Dummy unit.
    """
    DISABLE_KWARGS_CHECK = True

    def __init__(self, **kwargs):
        super(DummyUnit, self).__init__(DummyWorkflow())
        self.__dict__.update(kwargs)
