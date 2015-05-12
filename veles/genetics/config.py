# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Sep 8, 2014

Helpers for specifying paramters to optimize in config.

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


from logging import DEBUG, INFO
from multiprocessing import Process, Pipe, Value
import numpy
import sys
from zope.interface import implementer

from veles.compat import from_none
from veles.config import Config, root
from veles.cmdline import CommandLineBase
from veles.distributable import IDistributable
from veles.external.prettytable import PrettyTable
from veles.genetics.simple import Chromosome, Population
from veles.mutable import Bool
from veles.units import IUnit, Unit, nothing
from veles.workflow import Workflow, Repeater, NoMoreJobs
from veles.launcher import Launcher, filter_argv
from veles.plotting_units import AccumulatingPlotter
import veles.prng as prng


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    BrokenPipeError = OSError  # pylint: disable=W0622


class Tuneable(object):
    def __init__(self, default):
        self._path = None
        self._name = None
        self._addr = None
        self.default = default

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Tuneable's path must be a string (got %s)" % type(value))
        self._path = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Tuneable's name must be a string (got %s)" % type(value))
        self._name = value

    @property
    def full_name(self):
        return "%s.%s" % (self.path, self.name)

    @property
    def addr(self):
        return self._addr

    @addr.setter
    def addr(self, value):
        if not isinstance(value, tuple):
            raise TypeError(
                "Tuneable's addr must be a tuple (got %s)" % type(value))
        if len(value) != 2:
            raise ValueError(
                "Tuneable's addr must be of length = 2 (container, key)")
        # Check that the address is valid
        value[0][value[1]]
        self._addr = value

    def set(self, value):
        self.addr[0][self.addr[1]] = type(self.default)(value)

    def __ilshift__(self, value):
        self.set(value)

    def details(self):
        return "default: " + self.default

    def __str__(self):
        return "%s{%s}" % (type(self), self.details())

    def __repr__(self):
        return "%s: %s" % (self.full_name, str(self))


class Range(Tuneable):
    """Class for a tunable range.
    """
    def __init__(self, default, min_value, max_value):
        super(Range, self).__init__(default)
        self.min_value = min_value
        self.max_value = max_value

    def set(self, value):
        if value < self.min_value or value > self.max_value:
            raise ValueError(
                "[%s] Value is out of range [%s, %s]: %s" %
                (self.full_name, self.min_value, self.max_value, value))
        super(Range, self).set(value)

    def details(self):
        return "[%s, %s] (default: %s)" % (
            self.min_value, self.max_value, self.default)


def process_config(config, class_to_process, callback):
    """Applies callback to Config tree elements with the specified class.

    Parameters:
        config: instance of the Config object.
        class_to_process: class of the elements to which to apply the callback.
        callback: callback function with 3 arguments:
                  path: path in the Config tree (of type str) to this instance.
                  addr: tuple (container, key) pointing to this instance.
                  name: name of the parameter (of type str).
                  value: value of the parameter (of type class_to_process).
                  The return value is applied back.
    """
    _process_config(config.__path__, config.__dict__, class_to_process,
                    callback)


def _process_config(path, items, class_to_process, callback):
    if isinstance(items, dict):
        to_visit = items.items()
    else:
        to_visit = enumerate(items)
    to_process = {}
    for k, v in sorted(to_visit):
        if isinstance(v, Config):
            _process_config(v.__path__, v.__dict__, class_to_process, callback)
        elif isinstance(v, (dict, list, tuple)):
            _process_config("%s.%s" % (path, k), v, class_to_process, callback)
        elif isinstance(v, class_to_process):
            to_process[k] = v
    for k, v in sorted(to_process.items()):
        items[k] = callback(path, (items, k), k, v)


def fix_config(cfgroot):
    """Replaces all Tuneable values in Config tree with its defaults.

    Parameters:
        cfgroot: instance of the Config object.
    """
    return process_config(cfgroot, Tuneable, _fix_tuneable)


def _fix_tuneable(path, addr, name, value):
    return value.default


def print_config(cfgroot):
    for name, cfg in cfgroot.__content__.items():
        if name != "common":
            cfg.print_()


@implementer(IUnit, IDistributable)
class GeneticsContainer(Unit):
    """Unit which contains requested workflow for optimization.
    """
    def __init__(self, workflow, population, **kwargs):
        super(GeneticsContainer, self).__init__(workflow, **kwargs)
        self.population_ = population
        assert not self.is_standalone
        if self.is_slave:
            self._pipe = self.population_.job_connection[1]
            self._chromo = None
        else:
            self.pending_chromos = []
            self.retry_chromos = []
            self.scheduled_chromos = {}
            self._on_evaluation_finished = nothing
        self.max_fitness = -numpy.inf
        self.generation_evolved = Bool(False)

    def initialize(self, **kwargs):
        pass

    def run(self):
        """This will be executed on the slave.

        One chromosome at a time.
        """
        assert self.is_slave
        self.pipe.send(self.chromosome)
        try:
            self.chromosome.fitness = self.pipe.recv()  # blocks
        except:
            self.exception("Failed to receive the resulting fitness")
        else:
            self.gate_block <<= True

    @property
    def pipe(self):
        assert self.is_slave
        return self._pipe

    @property
    def _generation_evolved(self):
        if self.is_slave:
            return False
        return (len(self.scheduled_chromos) | len(self.retry_chromos) |
                len(self.pending_chromos)) == 0

    @property
    def on_evaluation_finished(self):
        assert self.is_master
        return self._on_evaluation_finished

    @on_evaluation_finished.setter
    def on_evaluation_finished(self, value):
        assert self.is_master
        self._on_evaluation_finished = value

    @property
    def chromosome(self):
        assert self.is_slave
        return self._chromo

    @chromosome.setter
    def chromosome(self, value):
        assert self.is_slave
        self._chromo = value

    @property
    def has_data_for_slave(self):
        return bool(len(self.retry_chromos) or len(self.pending_chromos))

    def generate_data_for_slave(self, slave):
        if slave.id in self.scheduled_chromos:
            # We do not support more than one job for a slave
            # Wait until the previous job finishes via apply_data_from_slave()
            self.warning("slave requested a new job, but the previous was not "
                         "completed => retry")
            idx = self.scheduled_chromos[slave.id]
            return self._chromo_by_idx(idx), idx
        try:
            idx = self.retry_chromos.pop()
        except IndexError:
            try:
                idx = self.pending_chromos.pop()
            except IndexError:
                raise NoMoreJobs()
        self.generation_evolved <<= False
        self.scheduled_chromos[slave.id] = idx
        self.info("Assigned chromosome %d to slave %s", idx, slave.id)
        return self._chromo_by_idx(idx), idx

    def apply_data_from_master(self, data):
        self.chromosome, idx = data
        assert self.chromosome is not None
        self.chromosome.population_ = self.population_
        self.info("Received chromosome #%d for evaluation", idx)
        self.gate_block <<= False

    def generate_data_for_master(self):
        self.debug("Sending to master fitness %.2f", self.chromosome.fitness)
        return self.chromosome.fitness

    def apply_data_from_slave(self, data, slave):
        idx = self.scheduled_chromos.pop(slave.id)
        chromo = self._chromo_by_idx(idx)
        chromo.fitness = data
        self.max_fitness = max(self.max_fitness, data)
        self.info("Got fitness %.2f for chromosome number %d", data, idx)
        if self._generation_evolved:
            self.info("Evaluated the entire population")
            self.generation_evolved <<= True
            self.on_evaluation_finished()  # pylint: disable=E1102

    def drop_slave(self, slave):
        try:
            idx = self.scheduled_chromos.pop(slave.id)
        except KeyError:
            self.warning("Dropped slave that had not received a job")
            return
        self.warning("Slave %s dropped, appending chromosome "
                     "number %d to the retry list", slave.id, idx)
        self.retry_chromos.append(idx)

    def enqueue_for_evaluation(self, chromo, idx):
        self.pending_chromos.append(idx)

    def _chromo_by_idx(self, idx):
        assert self.is_master  # slaves do not have the whole population
        return self.population_.chromosomes[idx]


class GeneticsWorkflow(Workflow):
    """Workflow which contains requested workflow for optimization.
    """
    def __init__(self, workflow, **kwargs):
        super(GeneticsWorkflow, self).__init__(workflow, **kwargs)

        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)

        self.population = kwargs["population"]
        self.container = GeneticsContainer(self, self.population)
        self.population.container = self.container
        self.container.link_from(self.repeater)

        self.plotter = AccumulatingPlotter(
            self, name="Genetic Optimization Max Fitness",
            plot_style="g-", redraw_plot=True, clear_plot=True)
        self.plotter.link_attrs(self.container, ("input", "max_fitness"))
        self.plotter.link_from(self.container)
        self.plotter.gate_skip = ~self.container.generation_evolved

        self.repeater.link_from(self.plotter)
        self.end_point.link_from(self.container)
        self.end_point.gate_block = ~self.container.gate_block
        self.plotter.gate_block = self.container.gate_block

    def initialize(self, **kwargs):
        super(GeneticsWorkflow, self).initialize(**kwargs)
        if self.is_master:
            self.population.evolve_on_master()

    @property
    def computing_power(self):
        avg_time = self.container.average_run_time
        if avg_time > 0:
            return 10000 / avg_time
        else:
            return 0

    def stop(self):
        self.container.gate_block <<= True
        super(GeneticsWorkflow, self).stop()


class ConfigChromosome(Chromosome):
    """Chromosome, based on Config tree's Tuneable elements.
    """
    def __init__(self, population,
                 size, min_values, max_values, accuracy, codes,
                 binary, numeric, rand):
        self.population_ = population
        self.fitness = None
        super(ConfigChromosome, self).__init__(
            size, min_values, max_values, accuracy, codes, binary, numeric,
            rand)

    def apply_config(self):
        for tune, val in zip(self.population_.registered_tunes_, self.numeric):
            tune <<= val

    def evaluate(self):
        self.apply_config()
        while self.fitness is None:
            self.fitness = self.evaluate_config()
        self.info("FITNESS = %.2f", self.fitness)

    def evaluate_config(self):
        """Evaluates current Config root.
        """
        fitness = Value('d', 0.0)
        p = Process(target=self.run_workflow, args=(fitness,))
        p.start()
        try:
            p.join()
        except KeyboardInterrupt as e:
            if p.is_alive():
                self.info("Giving the evaluator process a fair chance to die")
                p.join(1.0)
                if p.is_alive():
                    self.warning("Terminating the evaluator process")
                    p.terminate()
            raise from_none(e)
        if p.exitcode != 0:
            self.warning("Child process died with error code %d => "
                         "reevaluating", p.exitcode)
            return None
        return fitness.value

    def run_workflow(self, fitness):
        self.info("Will evaluate the following config:")
        root.common.disable.plotting = True
        root.common.disable.snapshotting = True
        root.common.disable.publishing = True
        print_config(self.population_.root_)
        self.population_.main_.run_module(self.population_.workflow_module_)
        fv = self.population_.main_.workflow.fitness
        if fv is not None:
            fitness.value = fv


class ConfigPopulation(Population):
    """Creates population based on Config tree's Tuneable elements.
    """
    def __init__(self, cfgroot, main, workflow_module, multi, size,
                 accuracy=0.00001, rand=prng.get()):
        """Constructor.

        Parameters:
            root: Config instance (NOTE: values of Tuneable class in it
                  will be changed during evolution).
            main: velescli Main instance.
            optimization_accuracy: float optimization accuracy.
        """
        self.root_ = cfgroot
        self.main_ = main
        self.workflow_module_ = workflow_module
        self.multi = multi
        self.container = None
        self.evaluations_pending = 0
        self.job_request_queue_ = None
        self.job_response_queue_ = None
        self.is_slave = None

        self.registered_tunes_ = []

        process_config(self.root_, Range, self.register_tune)
        if len(self.registered_tunes_) == 0:
            raise ValueError(
                "There are no tunable parameters in the configuration file. "
                "Wrap at least one into veles.genetics.Range class.")
        super(ConfigPopulation, self).__init__(
            ConfigChromosome,
            len(self.registered_tunes_),
            list(x.min_value for x in self.registered_tunes_),
            list(x.max_value for x in self.registered_tunes_),
            size, accuracy, rand)
        self.registered_tunes_.sort(
            key=lambda t: (str(type(t)), t.path, t.name))
        tpt = PrettyTable("path", "details", "class")
        tpt.align["path"] = 'l'
        tpt.align["details"] = 'l'
        for tune in self.registered_tunes_:
            tpt.add_row(tune.full_name, tune.details(), type(tune).__name__)
        self.info("Tuned parameters:\n%s", tpt)

    def register_tune(self, path, addr, name, value):
        value.path = path
        value.addr = addr
        value.name = name
        self.registered_tunes_.append(value)
        return value

    def log_statistics(self):
        self.info("#" * 80)
        self.info("Best config is:")
        best = self.chromosomes[
            numpy.argmax(x.fitness for x in self.chromosomes)]
        for tune, value in zip(self.registered_tunes_, best.numeric):
            tune <<= value
        print_config(self.root_)
        self.info("#" * 80)
        super(ConfigPopulation, self).log_statistics()
        self.info("#" * 80)

    def evaluate(self, callback):
        for chromo in self.chromosomes:
            chromo.population_ = self
        if not self.multi:
            return super(ConfigPopulation, self).evaluate(callback)
        self.container.on_evaluation_finished = callback
        for i, u in enumerate(self):
            if u.fitness is None:
                self.log(INFO if self.container.is_standalone else DEBUG,
                         "Enqueued for evaluation chromosome number %d "
                         "(%.2f%%)", i, 100.0 * i / len(self))
                self.container.enqueue_for_evaluation(u, i)

    def fix_argv_to_run_standalone(self, *argv_holders):
        """Forces standalone mode.

        Removes master-slave arguments from the command line.
        """
        for holder in argv_holders:
            self.debug("#" * 80)
            self.debug("%s.argv was %s", holder, holder.argv)
            holder.argv = ["-s", "-p", ""] + filter_argv(
                holder.argv, "-b", "--background", "-m", "--master-address")
            self.debug("%s.argv became %s", holder, holder.argv)
            self.debug("#" * 80)

    def job_process_main(self, parent_conn, child_conn):
        # Switch off genetics for the contained workflow launches
        self.fix_argv_to_run_standalone(sys, self.main_, CommandLineBase)
        self.main_.optimization = False
        child_conn.close()
        while True:
            try:
                chromo = parent_conn.recv()
            except KeyboardInterrupt:
                self.critical("KeyboardInterrupt")
                break
            if chromo is None:
                break
            try:
                chromo.population_ = self
                chromo.evaluate()
                parent_conn.send(chromo.fitness)
            except Exception as e:
                self.error("Failed to evaluate %s: %s", chromo, e)
                parent_conn.send(None)

    def evolve_multi(self):
        parser = Launcher.init_parser()
        args, _ = parser.parse_known_args()
        self.is_slave = bool(args.master_address.strip())
        if self.is_slave:
            # Fork before creating the GPU device
            self.job_connection = Pipe()
            self.job_process = Process(target=self.job_process_main,
                                       args=self.job_connection)
            self.job_process.start()
            self.job_connection[0].close()

            root.common.disable.plotting = True

        # Launch the container workflow
        self.main_.run_workflow(GeneticsWorkflow,
                                kwargs_load={"population": self})

        if self.is_slave:
            # Terminate the worker process
            try:
                self.job_connection[1].send(None)
            except BrokenPipeError:
                pass
            for conn in self.job_connection:
                conn.close()
            self.job_process.join()

    def evolve_on_master(self):
        super(ConfigPopulation, self).evolve()

    def evolve(self):
        if self.multi:
            self.evolve_multi()
        else:
            super(ConfigPopulation, self).evolve()

    def on_after_evolution_step(self):
        completed = super(ConfigPopulation, self).on_after_evolution_step()
        if completed and self.multi and not self.is_slave:
            # Stop master's workflow
            self.main_.workflow.stop()
        return completed
