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

Ensemble of machine learning algorithms - top level test workflow.

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


import json
import os
from tempfile import NamedTemporaryFile
from six import string_types
from zope.interface import implementer
from veles.ensemble.base_workflow import EnsembleWorkflowBase, \
    EnsembleModelManagerBase

from veles.units import IUnit


@implementer(IUnit)
class EnsembleTestManager(EnsembleModelManagerBase):
    def __init__(self, workflow, **kwargs):
        super(EnsembleTestManager, self).__init__(workflow, **kwargs)
        self.input_file = kwargs["input_file"]

    @property
    def input_file(self):
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        if not isinstance(value, string_types):
            raise TypeError(
                "input_file must be a string (got %s)" % type(value))
        self._input_file = value
        with open(self.input_file, "r") as fin:
            self._input_data = json.load(fin)
        self._results[:] = [None] * self.size

    @property
    def size(self):
        return len(self._input_data["models"])

    def initialize(self, **kwargs):
        super(EnsembleTestManager, self).initialize(**kwargs)
        if self.testing:
            self.warning("--test is ignored")

    def run(self):
        index = sum(1 for r in self.results if r is not None)
        model = self._input_data["models"][self._model_index]
        id_ = model["id"]
        log_id = model["log_id"]
        snapshot = model["Snapshot"]
        with NamedTemporaryFile(
                prefix="veles-ensemble-", suffix=".json", mode="r") as fin:
            argv = ["--test", "--result-file", fin.name, "--stealth",
                    "--log-id", self.launcher.log_id, "--snapshot",
                    self._to_snapshot_arg(id_, log_id, snapshot)] + \
                self._filtered_argv_ + ["root.common.disable.publishing=True"]
            try:
                self.info("Evaluating model %d / %d (#%d)...\n%s",
                          index + 1, self.size, self._model_index, "-" * 80)
                fin.seek(0, os.SEEK_SET)
                result = self._exec(argv, fin, "test")
                if result is None:
                    return
                self.results[index] = result
            finally:
                self._model_index += 1


class EnsembleTestWorkflow(EnsembleWorkflowBase):
    KWATTRS = set(EnsembleTestManager.KWATTRS)
    MANAGER_UNIT = EnsembleTestManager


def run(load, main, **kwargs):
    load(EnsembleTestWorkflow, **kwargs)
    main()
