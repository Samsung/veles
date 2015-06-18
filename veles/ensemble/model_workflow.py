# -*- coding: utf-8 -*-  # pylint: disable=C0302
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 11, 2015

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


import os
from tempfile import NamedTemporaryFile
from zope.interface import implementer

from veles.ensemble.base_workflow import EnsembleWorkflowBase, \
    EnsembleModelManagerBase
from veles.units import IUnit


@implementer(IUnit)
class EnsembleModelManager(EnsembleModelManagerBase):
    def __init__(self, workflow, **kwargs):
        super(EnsembleModelManager, self).__init__(workflow, **kwargs)
        self.size = kwargs["size"]
        self._train_ratio = kwargs["train_ratio"]

    @property
    def size(self):
        return len(self.results)

    @size.setter
    def size(self, value):
        if not isinstance(value, int):
            raise TypeError("size must be an integer (got %s)" % type(value))
        if value < 1:
            raise ValueError("size must be > 0 (got %d)" % value)
        self._results[:] = [None] * value

    @property
    def train_ratio(self):
        return self._train_ratio

    def initialize(self, **kwargs):
        super(EnsembleModelManager, self).initialize(**kwargs)
        if self.is_slave:
            self.size = 1
        if self.testing:
            raise ValueError(
                "Ensemble training is incompatibe with --test mode. Use "
                "--ensemble-test instead.")

    def run(self):
        index = sum(1 for r in self.results if r is not None)
        with NamedTemporaryFile(
                prefix="veles-ensemble-", suffix=".json", mode="r") as fin:
            argv = ["--result-file", fin.name, "--stealth", "--train-ratio",
                    str(self._train_ratio), "--log-id",
                    self.launcher.log_id] + self._filtered_argv_ + \
                   ["root.common.ensemble.model_index=%d" % self._model_index,
                    "root.common.ensemble.size=%d" % self.size,
                    "root.common.disable.publishing=True"]
            try:
                self.info("Training model %d / %d (#%d)...\n%s",
                          index + 1, self.size, self._model_index, "-" * 80)
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
            finally:
                self._model_index += 1


class EnsembleModelWorkflow(EnsembleWorkflowBase):
    KWATTRS = set(EnsembleModelManager.KWATTRS)
    MANAGER_UNIT = EnsembleModelManager


def run(load, main, **kwargs):
    load(EnsembleModelWorkflow, **kwargs)
    main()
