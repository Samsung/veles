# -*- coding: utf-8 -*-  # pylint: disable=C0302
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jub 11, 2015

Ensemble of machine learning algorithms - top level workflow.

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
from six import string_types
from zope.interface import implementer

from veles.paths import __root__
from veles.distributable import IDistributable
from veles.launcher import filter_argv
from veles.mutable import Bool
from veles.plumbing import Repeater
from veles.result_provider import IResultProvider
from veles.units import IUnit, Unit
from veles.workflow import Workflow


@implementer(IUnit, IDistributable, IResultProvider)
class EnsembleModelManager(Unit):
    def __init__(self, workflow, **kwargs):
        super(EnsembleModelManager, self).__init__(workflow, **kwargs)
        self._model_ = kwargs["model"]
        self.size = kwargs["size"]
        self._train_ratio = kwargs["train_ratio"]
        self._model_index = 0
        self._complete = Bool(lambda: None not in self.results)

    def init_unpickled(self):
        super(EnsembleModelManager, self).init_unpickled()
        self._pending_ = defaultdict(set)

    @property
    def complete(self):
        return self._complete

    @property
    def results(self):
        return self._results

    @property
    def size(self):
        return len(self.results)

    @size.setter
    def size(self, value):
        if not isinstance(value, int):
            raise TypeError("size must be an integer (got %s)" % type(value))
        if value < 1:
            raise ValueError("size must be > 0 (got %d)" % value)
        self._results = [None] * value

    @property
    def size_trained(self):
        return sum(1 for r in self.results if r is not None)

    @property
    def size_left(self):
        return self.size - self.size_trained - \
            sum(len(s) for s in self._pending_.values())

    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def model(self):
        return self._model_

    def initialize(self, **kwargs):
        if self.is_slave:
            self.size = 1

    def run(self):
        argv = filter_argv(
            sys.argv, "-l", "--listen-address", "-m", "--master-address", "-n",
            "--nodes", "-b", "--background", "-s", "--stealth",
            "--ensemble", "--slave-launch-transform", "--result-file")[1:]
        index = sum(1 for r in self.results if r is not None)
        with NamedTemporaryFile(
                prefix="veles-ensemble-", suffix=".json", mode="r") as fin:
            argv = ["--result-file", fin.name, "--stealth", "--train-ratio",
                    str(self._train_ratio), "--log-id",
                    self.launcher.log_id] + argv + \
                   ["root.common.ensemble.model_index=%d" % self._model_index,
                    "root.common.ensemble.size=%d" % self.size]
            self.info("Training model %d / %d (#%d)...\n%s",
                      index + 1, self.size, self._model_index, "-" * 80)
            self._model_index += 1
            train_result = self._exec(argv, fin, "train")
            if train_result is None:
                return
            try:
                id_ = train_result["id"]
                log_id = train_result["log_id"]
                snapshot = train_result["Snapshot"]
            except KeyError:
                self.error("Model #%d did not return a valid result",
                           self._model_index)
                return
            self.info("Evaluating model %d / %d (#%d)...\n%s",
                      index + 1, self.size, self._model_index, "-" * 80)
            argv = ["--test", "--snapshot", self._to_snapshot_arg(
                id_, log_id, snapshot)] + argv
            fin.seek(0, os.SEEK_SET)
            test_result = self._exec(argv, fin, "test")
            if test_result is None:
                return
            self.results[index] = train_result
            self.results[index].update(test_result)

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

    def get_metric_names(self):
        return {"models"}

    def get_metric_values(self):
        return {"models": self.results}

    def _exec(self, argv, fin, action):
        self.debug("exec: %s", " ".join(argv))
        __main__ = os.path.join(__root__, "veles", "__main__.py")
        env = {"PYTHONPATH": os.getenv("PYTHONPATH", __root__)}
        env.update(os.environ)
        if subprocess.call([sys.executable, __main__] + argv, env=env):
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


class EnsembleModelWorkflow(Workflow):
    KWATTRS = set(EnsembleModelManager.KWATTRS)

    def __init__(self, workflow, **kwargs):
        super(EnsembleModelWorkflow, self).__init__(workflow, **kwargs)
        self.ensemble = EnsembleModelManager(self, **kwargs)
        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)
        self.ensemble.link_from(self.repeater)
        self.repeater.link_from(self.ensemble)
        self.end_point.link_from(self.ensemble)
        self.end_point.gate_block = ~self.ensemble.complete
        self.repeater.gate_block = self.ensemble.complete


def run(load, main, **kwargs):
    load(EnsembleModelWorkflow, **kwargs)
    main()
