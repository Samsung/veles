"""
Created on Sep 8, 2014

Helpers for specifying paramters to optimize in config.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from logging import DEBUG, INFO
from multiprocessing import Process, Pipe, Value
import numpy
import sys
from zope.interface import implementer

from veles.config import Config, root
from veles.distributable import IDistributable
from veles.genetics import Chromosome, Population
from veles.units import IUnit, Unit
from veles.workflow import Workflow, Repeater
from veles.launcher import Launcher


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    BrokenPipeError = OSError  # pylint: disable=W0622


class Tuneable(object):
    def __init__(self):
        self.root = None
        self.name = None


class Tune(Tuneable):
    """Class for tunable range.
    """
    def __init__(self, defvle, minvle, maxvle):
        super(Tune, self).__init__()
        self.defvle = defvle
        self.minvle = minvle
        self.maxvle = maxvle


def process_config(cfgroot, class_to_process, callback):
    """Applies callback to Config tree elements with the specified class.

    Parameters:
        cfgroot: instance of the Config object.
        class_to_process: class of the elements on which to apply callback.
        callback: callback function with 3 arguments:
                  root: instance of the Config object (leaf of the tree).
                  name: name of the parameter (of type str).
                  value: value of the parameter (of type class_to_process).
    """
    kv = {}
    if isinstance(cfgroot, Config):
        arr = sorted(cfgroot.__dict__.items())
    elif isinstance(cfgroot, dict):
        arr = sorted(cfgroot.items())
    else:
        arr = enumerate(cfgroot)
    for k, v in arr:
        if isinstance(v, Config) or type(v) in (list, tuple, dict):
            process_config(v, class_to_process, callback)
        elif isinstance(v, class_to_process):
            kv[k] = v
    for k, v in sorted(kv.items()):
        callback(cfgroot, k, v)


def set_config_or_array(r, n, v):
    if isinstance(r, Config):
        setattr(r, n, v)
    else:
        r[n] = v


def fix_attr(r, n, v):
    set_config_or_array(r, n, v.defvle)


def fix_config(cfgroot):
    """Replaces all Tune values in Config tree with its defaults.

    Parameters:
        cfgroot: instance of the Config object.
    """
    return process_config(cfgroot, Tune, fix_attr)


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
            self._on_evaluation_finished = self.nothing

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
    def generation_evolved(self):
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

    def generate_data_for_slave(self, slave):
        if slave.id in self.scheduled_chromos:
            # We do not support more than one job for a slave
            # Wait until the previous job finishes via apply_data_from_slave()
            return False
        try:
            idx = self.retry_chromos.pop()
        except IndexError:
            try:
                idx = self.pending_chromos.pop()
            except IndexError:
                return False
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
        self.info("Got fitness %.2f for chromosome number %d", data, idx)
        if self.generation_evolved:
            self.debug("Evaluated everything, breeding season approaches...")
            self.on_evaluation_finished()

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

        self.repeater.link_from(self.container)
        self.end_point.link_from(self.container)
        self.end_point.gate_block = ~self.container.gate_block

    def initialize(self, **kwargs):
        super(GeneticsWorkflow, self).initialize(**kwargs)
        if self.is_master:
            self.population.evolve_on_master()

    @property
    def computing_power(self):
        avg_time = self.container.average_run_time
        if avg_time > 0:
            return 9999.99 / avg_time
        else:
            return 0


class ConfigChromosome(Chromosome):
    """Chromosome, based on Config tree's Tune elements.
    """
    def __init__(self, population,
                 size, minvles, maxvles, accuracy, codes,
                 binary, numeric):
        self.population_ = population
        self.fitness = None
        super(ConfigChromosome, self).__init__(
            size, minvles, maxvles, accuracy, codes, binary, numeric)

    def apply_config(self):
        for i, tune in enumerate(self.population_.registered_tunes_):
            set_config_or_array(tune.root, tune.name, self.numeric[i])

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
        except KeyboardInterrupt:
            if p.is_alive():
                self.info("Giving the evaluator process a fair chance to die")
                p.join(1.0)
                if p.is_alive():
                    self.warning("Terminating the evaluator process")
                    p.terminate()
            raise
        if p.exitcode != 0:
            self.warning("Child process died with error code %d => "
                         "reevaluating", p.exitcode)
            return None
        return fitness.value

    def run_workflow(self, fitness):
        self.info("Will evaluate the following config:")
        self.population_.root_.print_config()
        if self.population_.multi:
            self.population_.force_standalone()
        self.population_.main_.run_module(self.population_.workflow_module_)
        fv = self.population_.main_.workflow.fitness
        if fv is not None:
            fitness.value = fv


class ConfigPopulation(Population):
    """Creates population based on Config tree's Tune elements.
    """
    def __init__(self, cfgroot, main, workflow_module, multi, size,
                 accuracy=0.00001):
        """Constructor.

        Parameters:
            root: Config instance (NOTE: values of Tune class in it
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

        self.registered_tunes_ = []

        process_config(self.root_, Tune, self.register_tune)

        super(ConfigPopulation, self).__init__(
            ConfigChromosome,
            len(self.registered_tunes_),
            list(x.minvle for x in self.registered_tunes_),
            list(x.maxvle for x in self.registered_tunes_),
            size, accuracy)

    def register_tune(self, cfgroot, name, value):
        value.root = cfgroot
        value.name = name
        self.registered_tunes_.append(value)

    def log_statistics(self):
        self.info("#" * 80)
        self.info("Best config is:")
        best = self.chromosomes[
            numpy.argmax(x.fitness for x in self.chromosomes)]
        for i, tune in enumerate(self.registered_tunes_):
            set_config_or_array(tune.root, tune.name, best.numeric[i])
        self.root_.print_config()
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

    def force_standalone(self):
        """Forces standalone mode.

        Removes master-slave arguments from the command line.
        """
        self.info("#" * 80)
        self.info("sys.argv was %s", str(sys.argv))
        args = [sys.argv[0]]
        skip = True
        was_s = False
        i_genetics = -1
        for i, arg in enumerate(sys.argv):
            if skip:
                skip = False
                continue
            if arg == "-b":  # avoid double daemonization
                continue
            if arg == "-m":  # should be standalone
                skip = True
                continue
            if arg == "-s":
                was_s = True
            if arg.startswith("--optimize"):
                i_genetics = i
            args.append(arg)
        if not was_s:
            args.insert(i_genetics + 2, "-s")
        sys.argv = args
        self.info("sys.argv became %s", str(sys.argv))
        self.info("#" * 80)

    def job_process(self, parent_conn, child_conn):
        # Switch off genetics for the contained workflow launches
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
            except:
                self.error("Failed to evaluate %s", chromo)
                parent_conn.send(None)

    def evolve_multi(self):
        parser = Launcher.init_parser()
        args, _ = parser.parse_known_args()
        is_slave = bool(args.master_address.strip())
        if is_slave:
            # Fork before creating the OpenCL device
            self.job_connection = Pipe()
            self.job_process = Process(target=self.job_process,
                                       args=self.job_connection)
            self.job_process.start()
            self.job_connection[0].close()

        root.common.plotters_disabled = True
        # Launch the container workflow
        self.main_.run_workflow(GeneticsWorkflow,
                                kwargs_load={"population": self})

        if is_slave:
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
