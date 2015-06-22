# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 22, 2015.

Manual test for veles.plotting_units.AutoHistogramPlotter (workflow).

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


import numpy
from zope.interface import implementer

from veles import prng
from veles.config import root
from veles.memory import Array
from veles.mutable import Bool
from veles.plotting_units import AutoHistogramPlotter
from veles.plumbing import Repeater
from veles.units import Unit, IUnit
from veles.workflow import Workflow


@implementer(IUnit)
class Appender(Unit):
    def __init__(self, workflow, **kwargs):
        super(Appender, self).__init__(workflow, **kwargs)
        self.rand = prng.get()
        self.output = Array()
        self.complete = Bool()

    def initialize(self, **kwargs):
        pass

    def run(self):
        if not self.output:
            arr = numpy.zeros(0)
        else:
            arr = self.output.mem
        arr = numpy.concatenate((arr, [self.rand.normal()]))
        self.output.reset(arr)
        if input("Enter \"stop\" to finish: ") == "stop":
            self.complete <<= True


class TestAutoHistogramWorkflow(Workflow):
    def __init__(self, workflow, **kwargs):
        super(TestAutoHistogramWorkflow, self).__init__(workflow, **kwargs)
        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)
        self.appender = Appender(self)
        self.appender.link_from(self.repeater)
        self.plotter = AutoHistogramPlotter(self)
        self.plotter.link_from(self.appender)
        self.plotter.input = self.appender.output
        self.repeater.link_from(self.plotter)
        self.end_point.link_from(self.plotter)
        self.repeater.gate_block = self.appender.complete
        self.end_point.gate_block = ~self.appender.complete


def run(load, main):
    root.common.disable.plotting = False
    load(TestAutoHistogramWorkflow)
    main()
