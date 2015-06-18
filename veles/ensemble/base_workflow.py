# -*- coding: utf-8 -*-  # pylint: disable=C0302
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 17, 2015

Ensemble of machine learning algorithms - base classes.

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


from collections import defaultdict
import json
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from six import string_types, add_metaclass
from zope.interface import implementer

from veles.distributable import IDistributable
from veles.launcher import filter_argv
from veles.mutable import Bool
from veles.paths import __root__
from veles.plumbing import Repeater
from veles.result_provider import IResultProvider
from veles.units import Unit, UnitCommandLineArgumentsRegistry
from veles.workflow import Workflow


@implementer(IDistributable, IResultProvider)
@add_metaclass(UnitCommandLineArgumentsRegistry)
class EnsembleModelManagerBase(Unit):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(EnsembleModelManagerBase, self).__init__(workflow, **kwargs)
        self._model_ = kwargs["model"]
        self._model_index = 0
        self._results = []
        self._complete = Bool(lambda: None not in self.results)

    def init_unpickled(self):
        super(EnsembleModelManagerBase, self).init_unpickled()
        self._pending_ = defaultdict(set)
        self._filtered_argv_ = []

    @property
    def complete(self):
        return self._complete

    @property
    def results(self):
        return self._results

    @property
    def size_processed(self):
        return sum(1 for r in self.results if r is not None)

    @property
    def size_left(self):
        return self.size - self.size_processed - \
            sum(len(s) for s in self._pending_.values())

    @property
    def model(self):
        return self._model_

    def initialize(self, **kwargs):
        self._filtered_argv_[:] = filter_argv(
            self.argv, "-l", "--listen-address", "-m", "--master-address",
            "-n", "--nodes", "-b", "--background", "-s", "--stealth",
            "--ensemble-train", "--ensemble-test", "--slave-launch-transform",
            "--result-file", "--pdb-on-finish")

    def generate_data_for_master(self):
        return self._model_index - 1, self.results[0]

    def generate_data_for_slave(self, slave):
        for i, r in enumerate(self.results):
            if r is None and i not in self._pending_[slave]:
                self.info("Enqueued model #%d / %d to %s", i + 1, self.size,
                          slave.id)
                self._pending_[slave].add(i)
                self.has_data_for_slave = self.size_left > 0
                return i

    def apply_data_from_master(self, data):
        self._model_index = data
        self.results[0] = None

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            return
        model_index, result = data
        self._pending_[slave].remove(model_index)
        self._results[model_index] = result

    def drop_slave(self, slave):
        if slave in self._pending_:
            self._pending_[slave].clear()
            self.has_data_for_slave = self.size_left > 0

    def get_metric_names(self):
        return {"models"}

    def get_metric_values(self):
        return {"models": self.results}

    def _exec(self, argv, fin, action):
        __main__ = os.path.join(__root__, "veles", "__main__.py")
        argv = [sys.executable, __main__] + argv
        self.debug("exec: %s", " ".join(argv))
        env = {"PYTHONPATH": os.getenv("PYTHONPATH", __root__)}
        env.update(os.environ)
        if subprocess.call(argv, env=env):
            self.warning("Failed to %s model #%d", action, self._model_index)
            return
        try:
            return json.load(fin)
        except ValueError as e:
            fin.seek(0, os.SEEK_SET)
            with NamedTemporaryFile(
                    prefix="veles-ensemble-", suffix=".json", mode="w",
                    delete=False) as fout:
                fout.write(fin.read())
                self.warning("Failed to parse %s: %s", fout.name, e)

    @staticmethod
    def _to_snapshot_arg(id_, log_id, snapshot):
        if isinstance(snapshot, string_types):
            return snapshot
        # ODBC case
        return ("%s&" * 4 + "%s") % (
            snapshot["odbc"], snapshot["table"], id_, log_id, snapshot["name"])


class EnsembleWorkflowBase(Workflow):
    MANAGER_UNIT = None

    def __init__(self, workflow, **kwargs):
        super(EnsembleWorkflowBase, self).__init__(workflow, **kwargs)
        self.ensemble = \
            self.MANAGER_UNIT(self, **kwargs)  # pylint: disable=E1102
        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)
        self.ensemble.link_from(self.repeater)
        self.repeater.link_from(self.ensemble)
        self.end_point.link_from(self.ensemble)
        self.end_point.gate_block = ~self.ensemble.complete
        self.repeater.gate_block = self.ensemble.complete
