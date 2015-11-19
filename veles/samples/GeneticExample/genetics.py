# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Sep 21, 2015

Example of genetic optimization.

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
from __future__ import division

from zope.interface import implementer

from veles.config import root
from veles.result_provider import IResultProvider
from veles.units import IUnit, Unit
from veles.workflow import Workflow


@implementer(IUnit, IResultProvider)
class Optimizer(Unit):
    """Computes fitness value.
    """
    def __init__(self, workflow, **kwargs):
        super(Optimizer, self).__init__(workflow, **kwargs)
        self.fitness = 0.0
        # Create some attributes here.

    def initialize(self, **kwargs):
        # Allocate some resources here.
        pass

    def run(self):
        # Do the job here.
        x = root.test.x
        y = root.test.y
        value = (x - 0.33) ** 2 * (y - 0.27) ** 2

        # Assign the fitness (the more the better)
        self.fitness = -value  # looking for a minimum

    def get_metric_names(self):
        return {"EvaluationFitness"}

    def get_metric_values(self):
        return {"EvaluationFitness": self.fitness}


class TestWorkflow(Workflow):
    """Workflow for one run of fitness computation.
    """
    def __init__(self, workflow, **kwargs):
        super(TestWorkflow, self).__init__(workflow, **kwargs)
        self.optimizer = Optimizer(self).link_from(self.start_point)
        self.end_point.link_from(self.optimizer)


def run(load, main):
    """Entry point.
    """
    load(TestWorkflow)  # creates workflow or loads it from snapshot
    main()  # calls initialize and run on the workflow
