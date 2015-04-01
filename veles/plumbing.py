"""
Created on Apr 1, 2015

Service units to change the control flow.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from zope.interface import implementer

from veles.distributable import IDistributable, TriviallyDistributable
from veles.units import TrivialUnit, Unit, IUnit


class Repeater(TrivialUnit):
    """Completes a typical control flow cycle, usually joining the first unit
    with the last one.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "PLUMBING")
        kwargs["ignore_gate"] = True
        super(Repeater, self).__init__(workflow, **kwargs)

    def link_from(self, *args):
        super(Repeater, self).link_from(*args)
        if len(self.links_to) > 2:
            self.warning(
                "Repeater has more than 2 incoming links: %s. Are you sure?",
                tuple(self.links_to))


class UttermostPoint(TrivialUnit):
    hide_from_registry = True

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(UttermostPoint, self).__init__(workflow, **kwargs)


class StartPoint(UttermostPoint):
    """Workflow execution normally starts from this unit.
    """
    hide_from_registry = True

    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Start")
        super(StartPoint, self).__init__(workflow, **kwargs)

    @Unit.name.getter
    def name(self):
        if hasattr(self, "_workflow_") and self.workflow is not None:
            return "%s of %s" % (self._name, type(self.workflow).__name__)
        return Unit.name.fget(self)


class EndPoint(UttermostPoint):
    """Ends the pipeline execution, normally is the last unit in a workflow.
    """
    hide_from_registry = True

    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "End")
        super(EndPoint, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(EndPoint, self).init_unpickled()
        # on_workflow_finished() applies to Workflow's run time
        del Unit.timers[self.id]

    @Unit.name.getter
    def name(self):
        if hasattr(self, "_workflow_") and self.workflow is not None:
            return "%s of %s" % (self._name, type(self.workflow).__name__)
        return Unit.name.fget(self)

    def run(self):
        self.workflow.on_workflow_finished()

    def generate_data_for_master(self):
        return True

    def apply_data_from_slave(self, data, slave):
        if not self.gate_block:
            self.workflow.on_workflow_finished()


@implementer(IDistributable, IUnit)
class FireStarter(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(FireStarter, self).__init__(workflow, **kwargs)
        self._units = set(kwargs.get("units", tuple()))

    @property
    def units(self):
        return self._units

    def initialize(self, **kwargs):
        pass

    def run(self):
        for unit in self.units:
            unit.stopped = False
