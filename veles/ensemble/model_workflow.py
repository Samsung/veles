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


import json
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from zope.interface import implementer

from veles.paths import __root__
from veles.distributable import IDistributable
from veles.launcher import filter_argv
from veles.mutable import Bool
from veles.plumbing import Repeater
from veles.units import IUnit, Unit
from veles.workflow import Workflow


@implementer(IUnit, IDistributable)
class EnsembleModelManager(Unit):
    def __init__(self, workflow, **kwargs):
        super(EnsembleModelManager, self).__init__(workflow, **kwargs)
        self._model_ = kwargs["model"]
        self._size = kwargs["size"]
        self._train_ratio = kwargs["train_ratio"]
        self._aux_file = kwargs["aux_file"]
        self._results = []
        self._complete = Bool(lambda: len(self.results) == self.size)

    @property
    def complete(self):
        return self._complete

    @property
    def results(self):
        return self._results

    @property
    def size(self):
        return self._size

    def initialize(self, **kwargs):
        if not os.path.exists(self._aux_file):
            with open(self._aux_file, "w") as _:
                pass
        elif not os.access(self._aux_file, os.W_OK):
            raise ValueError("Cannot write to %s" % self._aux_file)

    def run(self):
        argv = filter_argv(
            sys.argv, "-l", "--listen-address", "-n", "--nodes",
            "-b", "--background", "-s", "--stealth", "--ensemble-stage",
            "--ensemble-definition", "--ensemble-aux-file",
            "--slave-launch-transform", "--result-file")[1:]
        index = len(self.results)
        with NamedTemporaryFile(
                prefix="veles-ensemble-", suffix=".json", mode="r") as fin:
            argv = ["--result-file", fin.name, "--stealth", "--train-ratio",
                    str(self._train_ratio), "--log-id",
                    self.launcher.log_id] + argv + \
                   ["root.common.ensemble.model_index=%d" % index,
                    "root.common.ensemble.size=%d" % self.size]
            self.info("Training model %d / %d...\n%s",
                      index + 1, self.size, "-" * 80)
            self.debug("%s", " ".join(argv))
            __main__ = os.path.join(__root__, "veles", "__main__.py")
            env = {"PYTHONPATH": os.getenv("PYTHONPATH", __root__)}
            env.update(os.environ)
            if subprocess.call([sys.executable, __main__] + argv, env=env):
                self.warning("Failed to train model #%d", index + 1)
                self._results.append(None)
                return
            try:
                self._results.append(json.load(fin))
            except ValueError as e:
                fin.seek(0, os.SEEK_SET)
                with NamedTemporaryFile(
                        prefix="veles-ensemble-", suffix=".json", mode="w",
                        delete=False) as fout:
                    fout.write(fin.read())
                    self.warning("Failed to parse %s: %s", fout.name, e)
                self._results.append(None)

    def stop(self):
        self.info("Dumping the results to %s...", self._aux_file)
        with open(self._aux_file, "w") as fout:
            json.dump(self._results, fout)

    def generate_data_for_master(self):
        pass

    def generate_data_for_slave(self, slave):
        pass

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


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
