"""
Created on Oct 1, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import sys
from zope.interface.declarations import implementer

from veles.workflow import Workflow, Repeater
from veles.zmq_loader import ZeroMQLoader
from veles.units import Unit, IUnit
from veles.distributable import IDistributable, TriviallyDistributable
from veles.mutable import Bool


@implementer(IUnit, IDistributable)
class Printer(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        super(Printer, self).__init__(workflow, **kwargs)
        self.exit = Bool()
        self.demand("input")

    def initialize(self, **kwargs):
        pass

    def run(self):
        print(self.input)
        sys.stdout.flush()
        if self.input == "exit":
            self.exit <<= True

    def stop(self):
        pass


class MastodonSampleWorkflow(Workflow):
    """
    Listens to texts coming from Mastodon Java clients and prints them.
    """

    def __init__(self, workflow, **kwargs):
        super(MastodonSampleWorkflow, self).__init__(workflow, **kwargs)
        if self.is_standalone:
            raise ValueError("Mastodon workflows can only be run in "
                             "master-slave mode")

        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)

        self.loader = ZeroMQLoader(self)
        self.loader.link_from(self.repeater)

        self.printer = Printer(self)
        self.printer.link_attrs(self.loader, ("input", "output"))
        self.printer.link_from(self.loader)

        self.repeater.link_from(self.printer)
        self.end_point.link_from(self.printer)
        self.end_point.gate_block = ~self.printer.exit


def run(load, main):
    load(MastodonSampleWorkflow)
    main()
