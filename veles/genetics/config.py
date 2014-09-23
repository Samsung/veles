"""
Created on Sep 8, 2014

Helpers for specifying paramters to optimize in config.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from multiprocessing import Process, Pipe, Value
import numpy
import sys
from zope.interface import implementer

from veles.config import Config
from veles.distributable import IDistributable
from veles.genetics import Chromosome, Population
from veles.units import IUnit, Unit
from veles.workflow import Workflow, Repeater


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


def process_config(root, class_to_process, callback):
    """Applies callback to Config tree elements with the specified class.

    Parameters:
        root: instance of the Config object.
        class_to_process: class of the elements on which to apply callback.
        callback: callback function with 3 arguments:
                  root: instance of the Config object (leaf of the tree).
                  name: name of the parameter (of type str).
                  value: value of the parameter (of type class_to_process).
    """
    kv = {}
    if isinstance(root, Config):
        arr = sorted(root.__dict__.items())
    elif isinstance(root, dict):
        arr = sorted(root.items())
    else:
        arr = enumerate(root)
    for k, v in arr:
        if isinstance(v, Config) or type(v) in (list, tuple, dict):
            process_config(v, class_to_process, callback)
        elif isinstance(v, class_to_process):
            kv[k] = v
    for k, v in sorted(kv.items()):
        callback(root, k, v)


def set_config_or_array(r, n, v):
    if isinstance(r, Config):
        setattr(r, n, v)
    else:
        r[n] = v


def fix_attr(r, n, v):
    set_config_or_array(r, n, v.defvle)


def fix_config(root):
    """Replaces all Tune values in Config tree with its defaults.

    Parameters:
        root: instance of the Config object.
    """
    return process_config(root, Tune, fix_attr)


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


@implementer(IUnit, IDistributable)
class GeneticsContainer(Unit):
    """Unit which contains requested workflow for optimization.
    """
    def __init__(self, workflow, **kwargs):
        super(GeneticsContainer, self).__init__(workflow, **kwargs)
        self.population_ = kwargs["population"]
        self.pending_chromos = []
        self.retry_chromos = []
        self.scheduled_chromos = {}
        self.chromo = None  # for slave only
        self.on_evaluation_finished = self.nothing

    def initialize(self, **kwargs):
        pass

    def run(self):
        """This will be executed on the slave.

        One chromosome at a time.
        """
        self.population_.job_connection[1].send(self.chromo)
        # Block until the resulting fitness is calculated
        self.chromo.fitness = self.population_.job_connection[1].recv()
        self.workflow.end_point.gate_block <<= False
        # Block until doRead() is fired
        self.gate_block <<= True

    def generate_data_for_slave(self, slave):
        if len(self.retry_chromos):
            chromo, idx = self.retry_chromos.pop(0)
        else:
            try:
                idx = self.pending_chromos.pop()
            except IndexError:
                return False
            chromo = self.population_.chromosomes[idx]
        self.scheduled_chromos[slave.id] = (chromo, idx)
        self.info("Sent chromosome %d to slave %s", idx, slave.id)
        return chromo, idx

    def apply_data_from_master(self, data):
        self.chromo, idx = data
        if self.chromo is not None:
            self.chromo.population_ = self.population_
            self.info("Received chromosome #%d for evaluation", idx)
        else:
            self.debug("Received None, None from master")
        self.workflow.end_point.gate_block <<= True
        self.gate_block <<= False

    def generate_data_for_master(self):
        if self.chromo is None:
            self.debug("Sending None to master")
            return None
        self.info("Sending to master fitness %.2f", self.chromo.fitness)
        return self.chromo.fitness

    def apply_data_from_slave(self, data, slave):
        chromo, idx = self.scheduled_chromos.pop(slave.id)
        chromo.fitness = data
        self.info("Got fitness %.2f for chromosome number %d", data, idx)
        if len(self.pending_chromos) == 0:
            self.debug("Evaluated everything, breeding season approaches...")
            self.on_evaluation_finished()

    def drop_slave(self, slave):
        try:
            chromo, idx = self.scheduled_chromos.pop(slave.id)
        except KeyError:
            self.warning("Dropped slave that had not received a job")
            return
        self.warning("Slave %s dropped, appending chromosome "
                     "number %d to the retry list", slave.id, idx)
        self.retry_chromos.append((chromo, idx))

    def enqueue_for_evaluation(self, chromo, idx):
        self.pending_chromos.append(idx)


class GeneticsWorkflow(Workflow):
    """Workflow which contains requested workflow for optimization.
    """
    def __init__(self, workflow, **kwargs):
        super(GeneticsWorkflow, self).__init__(workflow, **kwargs)

        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)

        self.population = kwargs["population"]
        self.container = GeneticsContainer(
            self, population=self.population)
        self.population.container = self.container
        self.container.link_from(self.repeater)

        self.repeater.link_from(self.container)
        self.end_point.link_from(self.container)
        self.end_point.gate_block <<= True

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


class ConfigPopulation(Population):
    """Creates population based on Config tree's Tune elements.
    """
    def __init__(self, root, main, workflow_module, multi,
                 optimization_accuracy=0.00001):
        """Constructor.

        Parameters:
            root: Config instance (NOTE: values of Tune class in it
                  will be changed during evolution).
            main: velescli Main instance.
            optimization_accuracy: float optimization accuracy.
        """
        self.root_ = root
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
            len(self.registered_tunes_),
            list(x.minvle for x in self.registered_tunes_),
            list(x.maxvle for x in self.registered_tunes_),
            optimization_accuracy=optimization_accuracy)

    def register_tune(self, root, name, value):
        value.root = root
        value.name = name
        self.registered_tunes_.append(value)

    def new_chromo(self, size, minvles, maxvles, accuracy, codes,
                   binary=None, numeric=None):
        return ConfigChromosome(
            self, size, minvles, maxvles, accuracy, codes, binary, numeric)

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
                self.info("Enqueued for evaluation chromosome number %d "
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
        # Fork before creating the OpenCL device
        self.job_connection = Pipe()
        job_process = Process(target=self.job_process,
                              args=self.job_connection)
        job_process.start()
        self.job_connection[0].close()

        # Launch the container workflow
        self.main_.run_workflow(GeneticsWorkflow,
                                kwargs_load={"population": self})

        # Terminate the worker process
        try:
            self.job_connection[1].send(None)
        except BrokenPipeError:
            pass
        for conn in self.job_connection:
            conn.close()
        job_process.join()

    def evolve_on_master(self):
        super(ConfigPopulation, self).evolve()

    def evolve(self):
        if self.multi:
            self.evolve_multi()
        else:
            super(ConfigPopulation, self).evolve()
