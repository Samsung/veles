"""
Created on Aug 6, 2013

Base class for workflows.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import hashlib
import numpy
import os
from six.moves import cPickle as pickle
import shutil
import sys
import tarfile
import tempfile
import yaml
import threading

import veles.benchmark as benchmark
from veles.config import root
import veles.formats as formats
from veles.units import Unit, OpenCLUnit, Repeater
import veles.external.pydot as pydot
from veles.external.prettytable import PrettyTable


class UttermostPoint(Unit):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "START_END")
        super(UttermostPoint, self).__init__(workflow, **kwargs)


class StartPoint(UttermostPoint):
    """Start point of a workflow execution.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Start")
        super(StartPoint, self).__init__(workflow, **kwargs)


class EndPoint(UttermostPoint):
    """End point with semaphore.

    Attributes:
        sem_: semaphore.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "End")
        super(EndPoint, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(EndPoint, self).init_unpickled()
        self.sem_ = threading.Semaphore(0)

    def run(self):
        self.sem_.release()

    def wait(self):
        self.sem_.acquire()

    def is_finished(self):
        b = self.sem_.acquire(False)
        if b:
            self.sem_.release()
        return b

    def generate_data_for_master(self):
        return True

    def apply_data_from_slave(self, data, slave):
        if not self.gate_block and not self.gate_skip:
            self.run()


class Workflow(Unit):
    """Base class for unit sets which are logically connected and belong to
    the same host.

    Attributes:
        start_point: start point.
        end_point: end point.
    """
    def __init__(self, workflow, **kwargs):
        self._plotters_are_enabled = not root.common.plotters_disabled
        super(Workflow, self).__init__(workflow, **kwargs)
        self.units = []
        self.start_point = StartPoint(self)
        self.end_point = EndPoint(self)
        self.thread_pool.register_on_shutdown(self.stop)

    def init_unpickled(self):
        super(Workflow, self).init_unpickled()
        self.master_pipeline_lock_ = threading.Lock()
        self.master_data_lock_ = threading.Lock()
        del(Unit.timers[self])

    def initialize(self):
        super(Workflow, self).initialize()
        if not self.is_slave:
            self.generate_graph()
        return self.start_point._initialize_dependent()

    def run(self):
        """Do the job here.

        In the child class:
            call the parent method at the end.
        """
        self.start_point.run_dependent()
        self.end_point.wait()
        self.print_stats()

    def add_ref(self, unit):
        if unit not in self.units:
            self.units.append(unit)

    def del_ref(self, unit):
        if unit in self.units:
            self.units.remove(unit)

    def lock_pipeline(self):
        """Locks master=>slave pipeline execution.
        """
        self.master_pipeline_lock_.acquire()

    def unlock_pipeline(self):
        """Unlocks master=>slave pipeline execution.
        """
        try:
            self.master_pipeline_lock_.release()
        except:
            self.warning("Double unlock in unlock_pipeline")

    def lock_data(self):
        """Locks master-slave data update.

        Read weights, apply gradients for example.
        """
        self.master_data_lock_.acquire()

    def unlock_data(self):
        """Unlocks master-slave data update.

        Read weights, apply gradients for example.
        """
        try:
            self.master_data_lock_.release()
        except:
            self.warning("Double unlock in unlock_data")

    def generate_data_for_master(self):
        data = []
        for unit in self.units:
            data.append(unit.generate_data_for_master())
        return data

    def generate_data_for_slave(self, slave):
        self.lock_pipeline()
        if self.is_finished():
            self.unlock_pipeline()
            return None
        data = []
        for unit in self.units:
            data.append(unit.generate_data_for_slave(slave))
        return data

    def apply_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        for i in range(0, len(data)):
            if data[i] is not None:
                self.units[i].apply_data_from_master(data[i])

    def apply_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        for i in range(len(self.units)):
            if data[i] is not None:
                self.units[i].apply_data_from_slave(data[i], slave)

    def drop_slave(self, slave):
        self.info("Job drop")
        for i in range(len(self.units)):
            self.units[i].drop_slave(slave)

    def request_job(self, slave):
        """
        Produces a new job, when a slave asks for it. Run by a master.
        """
        if self.is_finished():
            return None
        data = self.generate_data_for_slave(slave)
        return pickle.dumps(data) if data is not None else None

    def is_finished(self):
        return self.end_point.is_finished()

    @property
    def plotters_are_enabled(self):
        return self._plotters_are_enabled

    @plotters_are_enabled.setter
    def plotters_are_enabled(self, value):
        self._plotters_are_enabled = value

    def do_job(self, data):
        """
        Executes this workflow on the given source data. Run by a slave.
        """
        real_data = pickle.loads(data)
        self.apply_data_from_master(real_data)
        self.run()
        return pickle.dumps(self.generate_data_for_master())

    def apply_update(self, data, slave):
        """
        Harness the results of a slave's job. Run by a master.
        """
        real_data = pickle.loads(data)
        self.apply_data_from_slave(real_data, slave)

    def get_computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        return 0

    def stop(self):
        self.end_point.run()

    def generate_graph(self, filename=None, write_on_disk=True):
        g = pydot.Dot(graph_name="Workflow",
                      graph_type="digraph",
                      bgcolor="transparent")
        g.set_prog("circo")
        visited_units = set()
        boilerplate = set([self.start_point])
        while len(boilerplate) > 0:
            unit = boilerplate.pop()
            visited_units.add(unit)
            node = pydot.Node(hex(id(unit)))
            node.set("label", unit.name)
            node.set("shape", "rect")
            node.add_style("rounded")
            node.add_style("filled")
            color = Workflow.unit_group_colors.get(unit.view_group, "white")
            node.set("fillcolor", color)
            node.set("gradientangle", "90")
            g.add_node(node)
            for link in unit.links_to.keys():
                g.add_edge(pydot.Edge(hex(id(unit)), hex(id(link))))
                if link not in visited_units and link not in boilerplate:
                    boilerplate.add(link)
        if write_on_disk:
            if not filename:
                (_, filename) = tempfile.mkstemp(".png", "workflow_")
            self.info("Saving the workflow graph to %s", filename)
            g.write(filename, format='png')
        desc = g.to_string()
        self.debug("Graphviz workflow scheme:\n" + desc[:-1])
        return desc

    def print_stats(self, by_name=False, top_number=5):
        timers = {}
        for key, value in Unit.timers.items():
            uid = key.__class__.__name__ if not by_name else key.name()
            if id not in timers:
                timers[uid] = 0
            timers[uid] += value
        stats = sorted(timers.items(), key=lambda x: x[1], reverse=True)
        time_all = sum(timers.values())
        self.info("Unit run time statistics top:")
        for i in range(1, min(top_number, len(stats)) + 1):
            self.info("%d.  %s (%d%%)", i, stats[i - 1][0],
                      stats[i - 1][1] * 100 / time_all)

    unit_group_colors = {"PLOTTER": "gold",
                         "WORKER": "greenyellow",
                         "LOADER": "cyan",
                         "TRAINER": "coral",
                         "EVALUATOR": "plum",
                         "START_END": "lightgrey"}

    def checksum(self):
        sha1 = hashlib.sha1()
        with open(sys.argv[0]) as f:
            sha1.update(f.read().encode())
        return sha1.hexdigest()


class OpenCLWorkflow(OpenCLUnit, Workflow):
    """Base class for neural network workflows.

    Attributes:
        rpt: repeater.
        loader: loader unit.
        forward: list of the forward units.
        ev: evaluator unit.
        decision: decision unit.
        gd: list of the gradient descent units.
    """
    def __init__(self, workflow, **kwargs):
        super(OpenCLWorkflow, self).__init__(workflow, **kwargs)
        self.rpt = Repeater(self)
        self.loader = None
        self.forward = []
        self.ev = None
        self.decision = None
        self.gd = []
        self.power = None

    def initialize(self, device=None):
        super(OpenCLWorkflow, self).initialize()
        if device is not None:
            self.device = device
        for obj in self.forward:
            if obj is not None:
                obj.device = self.device
        if self.ev is not None:
            if type(self.ev) == list:
                for ev in self.ev:
                    if isinstance(ev, OpenCLUnit):
                        ev.device = self.device
            elif isinstance(self.ev, OpenCLUnit):
                self.ev.device = self.device
        for obj in self.gd:
            if obj is not None:
                obj.device = self.device

    def export(self, filename):
        """Exports workflow for use on DTV.
        """
        # create temporary folder
        tmppath = os.path.join(root.common.cache_dir, "saver_tmp")
        if not os.path.exists(tmppath):
            os.makedirs(tmppath)
        files_to_save = []
        dict_temp = {}
        variables_to_save = []
        # Go through units & save numpy array to binary file
        units_to_export = [self.loader]
        units_to_export.extend(self.forward)
        for i in range(len(units_to_export)):
            u = units_to_export[i]
            if u.exports is None:
                self.debug("%s continue" % u.__class__.__name__)
                continue
            variables = u.__getstate__()
            for key in variables:
                if key in u.exports:
                    self.debug("%s in attributes to export" % (key))
                    # Save numpy array to binary file
                    if type(getattr(u, key)) == formats.Vector and i >= 1:
                        for j in range(len(getattr(u, key).v.shape)):
                            name = key + "_shape_" + str(j)
                            self.info(name)
                            dict_temp[name] = getattr(u, key).v.shape[j]

                        link_to_numpy = "unit" + str(i - 1) + key + ".bin"

                        dict_temp['link_to_' + key] = link_to_numpy

                        files_to_save.append(
                            self._save_numpy_to_file(
                                getattr(u, key).v, link_to_numpy, tmppath))
                    else:
                        dict_temp[key] = getattr(u, key)
            temp__ = {}
            temp__[u.__class__.__name__] = dict_temp
            variables_to_save.append(temp__)
            dict_temp = {}

        # Save forward elements to yaml.
        yaml_name = 'default.yaml'
        self._save_to_yaml("%s/%s" % (tmppath, yaml_name), variables_to_save)
        # Compress archive
        tar = tarfile.open("%s.tar.gz" % (filename), "w:gz")
        tar.add("%s/%s" % (tmppath, yaml_name),
                arcname=yaml_name, recursive=False)
        for i in range(len(files_to_save)):
            tar.add("%s/%s" % (tmppath, files_to_save[i]),
                    arcname=files_to_save[i], recursive=False)
        tar.close()
        # delete temporary folder
        shutil.rmtree(tmppath)

    def _is_class_inside_object(self, obj_to_check):
        """Check that object is the class.
        Parameters:
            obj_to_check: object that should be checked.
        Returns:
            True if object is the class.
        """
        if isinstance(obj_to_check, (str, bytes, bool, int, float,
                                     list, tuple)) is False:
            return True
        return False

    def _save_to_yaml(self, yaml_name, to_yaml):
        """Print workflow to yaml-file.
        Parameters:
            yaml_name: filename to save.
        """
        stream = open(yaml_name, "w")
        for i in range(len(to_yaml)):
            yaml.dump(to_yaml[i], stream)
        stream.close()

    def _save_numpy_to_file(self, numpy_vector, numpy_vector_name, path):
        """Save numpy array to binary file.
        Parameters:
            numpy_vector: contains numpy array.
            numpy_vector_name: name of the binary file to save numpy array.
        """
        array_to_save = numpy.float32(numpy_vector.ravel())

        f = open("%s/%s" % (path, numpy_vector_name), "wb")
        f.write(array_to_save)
        f.close()
        return numpy_vector_name

    def get_computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        if not self.power:
            bench = benchmark.OpenCLBenchmark(self, device=self.device)
            self.power = bench.estimate()
            self.del_ref(bench)
            self.info("Computing power is %.6f", self.power)
        return self.power
