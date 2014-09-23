"""
Created on Sep 8, 2014

Helpers for specifying paramters to optimize in config.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from multiprocessing import Process, Queue
import numpy
import os
import queue
import threading
import time
from zope.interface import implementer

from veles.config import Config
from veles.distributable import IDistributable
from veles.genetics import Chromosome, Population
from veles.units import IUnit, Unit
from veles.workflow import Workflow, Repeater


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
    else:
        if type(root) == dict:
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
        super(ConfigChromosome, self).__init__(
            size, minvles, maxvles, accuracy, codes, binary, numeric)

    def apply_config(self):
        for i, tune in enumerate(self.population_.registered_tunes_):
            set_config_or_array(tune.root, tune.name, self.numeric[i])

    def evaluate(self):
        self.apply_config()

        while True:
            self.fitness = self.evaluate_config()
            if self.fitness is None:
                self.warning("Subprocess returned invalid fitness, "
                             "will reevaluate it in 30 seconds")
                time.sleep(30)
                self.warning("Will reevaluate now")
                continue
            break

        self.info("FITNESS = %.2f", self.fitness)

    def evaluate_config(self):
        """Evaluates current Config root.
        """
        q = Queue()
        p = Process(target=self.run_workflow, args=(q,))
        p.start()
        fitness = q.get()
        p.join()
        return fitness

    def run_workflow(self, q):
        import logging
        logging.basicConfig(level=logging.INFO)
        self.info("Will evaluate the following config:")
        self.population_.root_.print_config()
        if self.population_.multi:
            self.population_.force_standalone()
        self.population_.main_.run_module(self.population_.workflow_module_)
        fitness = self.population_.main_.workflow.fitness
        q.put(fitness)


@implementer(IUnit, IDistributable)
class GeneticsContainer(Unit):
    """Unit which contains requested workflow for optimization.
    """
    def __init__(self, workflow, **kwargs):
        super(GeneticsContainer, self).__init__(workflow, **kwargs)
        self.population_ = kwargs["population"]
        self.queue_ = Queue(1)
        self.retry_list = []
        self.chromo = None  # for slave only
        self.scheduled_chromos = {}
        self.thread_to_start = kwargs["thread_to_start"]
        self.last_exec_time = 86400

    def initialize(self, **kwargs):
        pass

    def run(self):
        """This will be executed on the slave.

        One chromosome at a time.
        """
        if self.is_master:
            raise ValueError("Should not be called in master mode")
        if self.chromo is not None:
            t0 = time.time()
            self.population_.job_request_queue_.put(self.chromo)
            self.chromo.fitness = self.population_.job_response_queue_.get()
            self.last_exec_time = time.time() - t0
        else:
            self.info("No job yet, sleeping for 5 seconds")
            time.sleep(5)
        self.workflow.end_point.gate_block <<= False
        self.gate_block <<= True

    def generate_data_for_slave(self, slave):
        if self.thread_to_start is not None:
            self.thread_to_start.start()
            self.thread_to_start = None
        if len(self.retry_list):
            chromo, idx = self.retry_list.pop(0)
        else:
            try:
                idx = self.queue_.get_nowait()
            except queue.Empty:
                self.debug("No job yet, sending None, None to slave %s",
                           slave.id)
                return None, None
            chromo = self.population_.chromosomes[idx]
        self.scheduled_chromos[slave.id] = (chromo, idx)
        self.info("Sent chromosome %d to slave %s", idx, slave.id)
        return chromo, idx

    def apply_data_from_master(self, data):
        self.chromo, idx = data
        if self.chromo is not None:
            self.chromo.population_ = self.population_
            self.info("Received for evaluation chromosome number %d", idx)
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
        with self.population_.lock_:
            self.population_.evaluations_pending -= 1

    def drop_slave(self, slave):
        try:
            chromo, idx = self.scheduled_chromos.pop(slave.id)
        except KeyError:
            self.warning("Dropped slave that has not received a job")
            return
        self.warning("Slave %s dropped, appending chromosome "
                     "number %d to the retry list", slave.id, idx)
        self.retry_list.append((chromo, idx))

    def enqueue_for_evaluation(self, chromo, idx):
        self.queue_.put(idx)

    def wait_for_evaluation(self):
        # TODO(a.kazantsev): implement properly.
        while True:
            with self.population_.lock_:
                if self.population_.evaluations_pending <= 0:
                    break
            time.sleep(0.5)
        self.debug("Evaluated everything, breeding season approaches...")


class GeneticsWorkflow(Workflow):
    """Workflow which contains requested workflow for optimization.
    """
    def __init__(self, workflow, **kwargs):
        super(GeneticsWorkflow, self).__init__(workflow, **kwargs)

        self.repeater = Repeater(self)
        self.repeater.link_from(self.start_point)

        population = kwargs["population"]
        self.container = GeneticsContainer(
            self, population=population,
            thread_to_start=kwargs["thread_to_start"])
        population.container = self.container
        self.container.link_from(self.repeater)

        self.repeater.link_from(self.container)
        self.end_point.link_from(self.container)
        self.end_point.gate_block <<= True

    @property
    def computing_power(self):
        return 9999.99 / self.container.last_exec_time


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
        self.lock_ = threading.Lock()
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

    def evaluate(self):
        for chromo in self.chromosomes:
            chromo.population_ = self
        if not self.multi:
            return super(ConfigPopulation, self).evaluate()
        with self.lock_:
            self.evaluations_pending = 0
        for i, u in enumerate(self):
            if u.fitness is None:
                self.info("Enqueued for evaluation chromosome number %d "
                          "(%.2f%%)", i, 100.0 * i / len(self))
                with self.lock_:
                    self.evaluations_pending += 1
                self.container.enqueue_for_evaluation(u, i)
        self.container.wait_for_evaluation()

    def force_standalone(self):
        """Forces standalone mode.

        Removes master-slave arguments from the command line.
        """
        self.info("#" * 80)
        import sys
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
            if arg.startswith("--genetics"):
                i_genetics = i
            args.append(arg)
        if not was_s:
            args.insert(i_genetics + 2, "-s")
        sys.argv = args
        self.info("sys.argv became %s", str(sys.argv))
        self.info("#" * 80)

    def job_process(self, request_queue, response_queue):
        try:
            self._job_process(request_queue, response_queue)
        except Exception as e:
            self.error("Exception occured while processing the job, "
                       "will exit the worker process, reason is: %s",
                       str(e))
            os._exit(1)

    def _job_process(self, request_queue, response_queue):
        # Switch off genetics for the contained workflow launches
        self.main_.genetics = False
        while True:
            chromo = request_queue.get()
            if chromo is None:
                break
            chromo.population_ = self
            chromo.evaluate()
            response_queue.put(chromo.fitness)

    def evolution(self):
        try:
            self._evolution()
        except Exception as e:
            self.error("Exception occured while doing the evolution, "
                       "will exit the main process, reason is: %s",
                       str(e))
            os._exit(1)

    def _evolution(self):
        if self.multi:
            # Fork before creating the twisted reactor
            self.job_request_queue_ = Queue()
            self.job_response_queue_ = Queue()
            job_process = Process(
                target=self.job_process,
                args=(self.job_request_queue_, self.job_response_queue_))
            job_process.start()
            # Launch thread for evolution,
            # thread will be started in the container workflow
            thread = threading.Thread(
                target=super(ConfigPopulation, self).evolution, args=())
            # Launch the container workflow
            self.main_.run_workflow(
                GeneticsWorkflow,
                kwargs_load={"population": self, "thread_to_start": thread})
            if thread.is_alive():  # it will not be started on slave
                thread.join()
            self.job_request_queue_.put(None)
            job_process.join()
        else:
            super(ConfigPopulation, self).evolution()
