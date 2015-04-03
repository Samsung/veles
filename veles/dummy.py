"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Oct 31, 2014

Dummy units for tests and benchmarks.

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
