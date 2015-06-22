# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 18, 2015

Model parameters optimization - top level workflow.

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

import copy
import json
import os
from six import add_metaclass
import sys
import subprocess
from tempfile import NamedTemporaryFile
from zope.interface import implementer
from veles import prng, __root__
from veles.compat import from_none

from veles.config import root
from veles.distributable import IDistributable
from veles.genetics.config import process_config, Range, print_config, \
    ConfigChromosome, ConfigPopulation, GeneticsJSONEncoder
from veles.launcher import filter_argv
from veles.mutable import Bool
from veles.pickle2 import best_protocol, pickle
from veles.plotting_units import AccumulatingPlotter
from veles.plumbing import Repeater
from veles.result_provider import IResultProvider
from veles.units import IUnit, UnitCommandLineArgumentsRegistry, Unit
from veles.workflow import Workflow


class EvaluationError(Exception):
    pass


@implementer(IUnit, IDistributable, IResultProvider)
@add_metaclass(UnitCommandLineArgumentsRegistry)
class GeneticsOptimizer(Unit):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(GeneticsOptimizer, self).__init__(workflow, **kwargs)
        self._model_ = kwargs["model"]
        self._config = copy.deepcopy(kwargs.get("config", root))
        if "config" not in kwargs:
            del self.config.common
        self.plotters_are_disabled = kwargs.get(
            "plotters_are_disabled", root.common.genetics.disable.plotting)
        self._tuneables = []
        process_config(self.config, Range, self._add_tuneable)
        if len(self.tuneables) == 0:
            raise ValueError(
                "There are no tunable parameters in the supplied configuration"
                " %s. Wrap at least one into veles.genetics.Range class." %
                self.config.__path__)
        self._chromosome_index = 0
        self.generation_changed = Bool()
        if self.is_slave:
            self.complete = Bool()
            return
        self._population = ConfigPopulation(
            lambda *a, **k: ConfigChromosome(self, *a, **k),
            len(self.tuneables),
            [x.min_value for x in self.tuneables],
            [x.max_value for x in self.tuneables],
            kwargs["size"], rand=kwargs.get("rand", prng.get()),
            max_generations=kwargs.get("generations"))
        self.population.on_generation_changed_callback = \
            self._set_generation_changed
        self._best_config = ""  # actual type is veles.config.Config
        self.complete = ~self.population.improved

    def init_unpickled(self):
        super(GeneticsOptimizer, self).init_unpickled()
        self._filtered_argv_ = []
        self._pending_ = defaultdict(set)

    @property
    def population(self):
        return self._population

    @property
    def best(self):
        assert not self.is_slave
        return self.population[0]

    @property
    def tuneables(self):
        return self._tuneables

    @property
    def config(self):
        return self._config

    @property
    def size(self):
        return self.population.size

    @property
    def pending_size(self):
        return sum(len(s) for s in self._pending_.values())

    @property
    def max_fitness(self):
        return self.population.best_fit or 0

    @property
    def avg_fitness(self):
        return self.population.average_fit or 0

    def initialize(self, **kwargs):
        self._filtered_argv_[:] = filter_argv(
            self.argv, "-l", "--listen-address", "-m", "--master-address",
            "-n", "--nodes", "-b", "--background", "-s", "--stealth",
            "--optimize", "--slave-launch-transform", "--result-file",
            "--pdb-on-finish")

    def run(self):
        self.generation_changed <<= False
        self.info("Evaluating chromosome #%d...", self._chromosome_index)
        self.population.evaluate(self._chromosome_index)
        self._chromosome_index += 1
        if self.is_slave:
            self.complete <<= True

    def stop(self):
        if self.is_slave:
            return
        self.info("Best fitness: %s", self.best.fitness)
        self.info("Best snapshot: %s", self.best.snapshot)
        self.info("Best configuration")
        print_config(self.best.config)

    def get_metric_names(self):
        return {"Fitness", "Best configuration", "Generation"}

    def get_metric_values(self):
        return {"Fitness": self.best.fitness,
                "Best configuration": self.best.config,
                "Snapshot": self.best.snapshot,
                "Generation": self.population.generation}

    def generate_data_for_master(self):
        index = self._chromosome_index - 1
        chromo = self.population[index]
        return index, (chromo.fitness, chromo.config, chromo.snapshot)

    def generate_data_for_slave(self, slave):
        self._update_has_more_data_for_slave()
        self.generation_changed <<= False
        for index in range(len(self.population)):
            if self.population[index].fitness is not None:
                continue
            if not any(index in s for s in self._pending_):
                self._pending_[slave].add(index)
                return self.population, index

    def apply_data_from_master(self, data):
        self._population, self._chromosome_index = data
        self.population[self._chromosome_index].unit = self
        # Prevent from running Population.update()
        for index, chromo in enumerate(self.population):
            if index != self._chromosome_index:
                chromo.fitness = None
        self.complete <<= False

    def apply_data_from_slave(self, data, slave):
        index, (fitness, config, snapshot) = data
        if index not in self._pending_[slave]:
            self.warning("No such job was given: %d", index)
            return
        chromosome = self.population[index]
        chromosome.fitness = fitness
        chromosome.config = config
        chromosome.snapshot = snapshot
        self.info("Chromosome #%d was evaluated to %s", index, fitness)
        self.population.update()
        self._pending_[slave].remove(index)

    def drop_slave(self, slave):
        if slave in self._pending_:
            self._pending_[slave].clear()
            self._update_has_more_data_for_slave()

    def evaluate(self, chromo):
        for tune, val in zip(self.tuneables, chromo.numeric):
            tune <<= val
        chromo.config = copy.deepcopy(self.config)
        with NamedTemporaryFile(mode="wb", prefix="veles-optimization-config-",
                                suffix=".%d.pickle" % best_protocol) as fcfg:
            pickle.dump(self.config, fcfg)
            fcfg.flush()
            with NamedTemporaryFile(
                    mode="r", prefix="veles-optimization-result-",
                    suffix=".%d.pickle" % best_protocol) as fres:
                argv = ["--result-file", fres.name, "--stealth", "--log-id",
                        self.launcher.log_id] + self._filtered_argv_ + \
                    ["root.common.disable.publishing=True"]
                if self.plotters_are_disabled:
                    argv = ["-p", ""] + argv
                i = -1
                while "=" in argv[i]:
                    i -= 1
                argv[i] = fcfg.name
                result = self._exec(argv, fres)
                if result is None:
                    raise EvaluationError()
        try:
            chromo.fitness = result["EvaluationFitness"]
        except KeyError:
            raise from_none(EvaluationError(
                "Failed to find \"EvaluationFitness\" in the evaluation "
                "results"))
        chromo.snapshot = result.get("Snapshot")
        self.info("Chromosome #%d was evaluated to %f", self._chromosome_index,
                  chromo.fitness)

    def _update_has_more_data_for_slave(self):
        self.has_data_for_slave = \
            self.pending_size < self.population.pending_size

    def _add_tuneable(self, path, addr, name, value):
        value.path = path
        value.addr = addr
        value.name = name
        self.tuneables.append(value)
        return value

    def _exec(self, argv, fin):
        __main__ = os.path.join(__root__, "veles", "__main__.py")
        argv = [sys.executable, __main__] + argv
        self.debug("exec: %s", " ".join(argv))
        env = {"PYTHONPATH": os.getenv("PYTHONPATH", __root__)}
        env.update(os.environ)
        if subprocess.call(argv, env=env):
            self.error("Failed to evaluate chromosome #%d",
                       self._chromosome_index)
            return
        try:
            return json.load(fin)
        except ValueError as e:
            fin.seek(0, os.SEEK_SET)
            with NamedTemporaryFile(
                    prefix="veles-optimization-", suffix=".json", mode="w",
                    delete=False) as fout:
                fout.write(fin.read())
                self.error("Failed to parse %s: %s", fout.name, e)

    def _set_generation_changed(self):
        self.generation_changed <<= True
        # That's right, I do mean it
        #    old          new
        # |--------|----------------|
        #         size
        # - 1 because it will be incremented in run()
        self._chromosome_index = self.size - 1


class OptimizationWorkflow(Workflow):
    KWATTRS = set(GeneticsOptimizer.KWATTRS)

    def __init__(self, workflow, **kwargs):
        super(OptimizationWorkflow, self).__init__(workflow, **kwargs)
        self.optimizer = GeneticsOptimizer(self, **kwargs)
        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)
        self.optimizer.link_from(self.repeater)

        self.plotter_max = AccumulatingPlotter(
            self, name="Genetic Optimization Fitness",
            plot_style="g-", redraw_plot=False, clear_plot=True)
        if self.plotters_are_enabled:
            self.plotter_max.link_attrs(self.optimizer,
                                        ("input", "max_fitness"))
        else:
            self.plotter_max.input = 0
        self.plotter_max.link_from(self.optimizer)
        self.plotter_max.gate_skip = ~self.optimizer.generation_changed

        self.plotter_avg = AccumulatingPlotter(
            self, name="Genetic Optimization Fitness",
            plot_style="b-", redraw_plot=True)
        if self.plotters_are_enabled:
            self.plotter_avg.link_attrs(self.optimizer,
                                        ("input", "avg_fitness"))
        else:
            self.plotter_avg.input = 0
        self.plotter_avg.link_from(self.plotter_max)
        self.plotter_avg.gate_skip = ~self.optimizer.generation_changed

        self.repeater.link_from(self.plotter_avg)
        self.end_point.link_from(self.plotter_avg)
        self.end_point.gate_block = ~self.optimizer.complete
        self.repeater.gate_block = self.optimizer.complete
        self.json_encoder = GeneticsJSONEncoder


def run(load, main, **kwargs):
    load(OptimizationWorkflow, **kwargs)
    main()
