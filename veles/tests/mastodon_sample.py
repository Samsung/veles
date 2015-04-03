"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Oct 1, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
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
