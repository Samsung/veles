"""
Created on Aug 6, 2013

Base class for workflows.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import os
import pickle
import shutil
import tarfile
import yaml

import config
import formats
import units
import benchmark
import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile

class Workflow(units.Unit):
    """Base class for workflows.

    Attributes:
        start_point: start point.
        end_point: end point.
    """
    def __init__(self):
        super(Workflow, self).__init__()
        self.start_point = units.Unit()
        self.end_point = units.EndPoint()

    def run(self):
        """Do the job here.

        In the child class:
            call the parent method at the end.
        """
        self.generate_graph()
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()

    def generate_data_for_master(self):
        data = self.start_point.generate_data_for_master_recursively()
        return pickle.dumps(data)

    def generate_data_for_slave(self):
        data = self.start_point.generate_data_for_slave_recursively()
        return pickle.dumps(data)

    def apply_data_from_master(self, data):
        real_data = pickle.loads(data)
        self.start_point.apply_data_from_master_recursively(real_data)

    def apply_data_from_slave(self, data):
        real_data = pickle.loads(data)
        self.apply_data_from_slave_recursively(real_data)

    def request_job(self):
        """
        Produces a new job, when a slave asks for it. Run by a master.
        """
        return self.generate_data_for_slave()

    def do_job(self, data):
        """
        Executes this workflow on the given source data. Run by a slave.
        """
        self.apply_data_from_master(data)
        self.run()
        return self.generate_data_for_master()

    def apply_update(self, data):
        """
        Harness the results of a slave's job. Run by a master.
        """
        self.apply_data_from_slave(data)

    def get_computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        return 0

    def generate_graph(self, filename=None):
        g = pydot.Dot(graph_name="Workflow",
                      graph_type="digraph",
                      mindist="0.1")
        g.set_prog("circo")
        visited_units = set()
        boilerplate = set([self.start_point])
        while len(boilerplate) > 0:
            unit = boilerplate.pop()
            visited_units.add(unit)
            node = pydot.Node(hex(id(unit)))
            node.set("label", unit.name())
            node.set("shape", "rect")
            node.add_style("rounded")
            g.add_node(node)
            for link in unit.links_to.keys():
                g.add_edge(pydot.Edge(hex(id(unit)), hex(id(link))))
                if link not in visited_units and link not in boilerplate:
                    boilerplate.add(link)
        if not filename:
            (_, filename) = tempfile.mkstemp(".png", "workflow_")
        g.write(filename, format='png')
        return g.to_string()


class OpenCLWorkflow(units.OpenCLUnit, Workflow):
    """Base class for neural network workflows.

    Attributes:
        rpt: repeater.
        loader: loader unit.
        forward: list of the forward units.
        ev: evaluator unit.
        decision: decision unit.
        gd: list of the gradient descent units.
    """
    def __init__(self, device=None):
        super(OpenCLWorkflow, self).__init__(device=device)
        self.rpt = units.Repeater()
        self.loader = None
        self.forward = []
        self.ev = None
        self.decision = None
        self.gd = []
        self.power = None

    def initialize(self, device=None):
        super(OpenCLWorkflow, self).initialize()
        if device != None:
            self.device = device
        for obj in self.forward:
            if obj != None:
                obj.device = self.device
        if self.ev != None:
            if type(self.ev) == list:
                for ev in self.ev:
                    if isinstance(ev, units.OpenCLUnit):
                        ev.device = self.device
            elif isinstance(self.ev, units.OpenCLUnit):
                self.ev.device = self.device
        for obj in self.gd:
            if obj != None:
                obj.device = self.device

    def export(self, filename):
        """Exports workflow for use on DTV.
        """
        # create temporary folder
        tmppath = "%s/saver_tmp" % (config.cache_dir)
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
            if u.exports == None:
                self.log().debug("%s continue" % u.__class__.__name__)
                continue
            variables = u.__getstate__()
            for key in variables:
                if key in u.exports:
                    self.log().debug("%s in attributes to export" % (key))
                    # Save numpy array to binary file
                    if type(getattr(u, key)) == formats.Vector and i >= 1:
                        for j in range(len(getattr(u, key).v.shape)):
                            name = key + "_shape_" + str(j)
                            self.log().info(name)
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
                                     list, tuple)) == False:
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
            bench = benchmark.OpenCLBenchmark()
            self.power = bench.estimate()
        return self.power
