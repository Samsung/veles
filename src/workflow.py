"""
Created on Aug 6, 2013

Base class for workflows.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units


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
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


import os
import yaml
import formats
import config
import tarfile
import numpy
import shutil


class NNWorkflow(units.OpenCLUnit, Workflow):
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
        super(NNWorkflow, self).__init__(device=device)
        self.rpt = units.Repeater()
        self.loader = None
        self.forward = []
        self.ev = None
        self.decision = None
        self.gd = []

    def initialize(self, device=None):
        super(NNWorkflow, self).initialize()
        if device != None:
            self.device = device
        for obj in self.forward:
            if obj != None:
                obj.device = self.device
        if self.ev != None:
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
        variables_to_save = []
        # Go through forward elements & save numpy array to binary file &
        # delete some variables
        for i in range(len(self.forward)):
            variables = self.forward[i].__getstate__()
            for key in variables:
                if key in self.forward[i].attributes_to_save:
                    self.log().debug("%d) %s in attributes to save" % (i, key))
                    # Save numpy array to binary file
                    if type(getattr(self.forward[i], key)) == formats.Vector:
                        for j in range(len(getattr(self.forward[i], key).v.shape)):
                            name = key + "_shape_" + str(j)
                            setattr(self.forward[i], name,
                                getattr(self.forward[i], key).v.shape[j])

                        files_to_save.append(
                            self._save_numpy_to_file(
                                getattr(self.forward[i], key).v,
                                key, i, tmppath))
                        delattr(self.forward[i], key)
                else:
                    delattr(self.forward[i], key)
            variables = self.forward[i].__getstate__()
            variables_to_save.append(variables)

        # Save forward elements to yaml.
        yaml_name = 'default.yaml'
        self._save_to_yaml("%s/%s" % (tmppath, yaml_name))
        # Compress archive
        tar = tarfile.open("%s.tar.gz" % (filename), "w:gz")
        tar.add("%s/%s" % (tmppath, yaml_name),
                arcname=yaml_name, recursive=False)
        for i in range(len(files_to_save)):
            tar.add("%s/%s" % (tmppath, files_to_save[i]),
                    arcname=files_to_save[i], recursive=False)
        tar.close()
#         # delete temporary folder
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

    def _save_to_yaml(self, yaml_name):
        """Print workflow to yaml-file.
        Parameters:
            yaml_name: filename to save.
        """
        stream = open(yaml_name, "w")

        for i in range(len(self.forward)):
            to_print = {}
            cls = self.forward[i].__class__
            self.log().debug("Saving " + cls.__name__ + "...")
            cls.yaml_dumper.add_representer(cls, cls.to_yaml)
            cls_name = self.forward[i].__class__.__name__
            # TODO(EBulychev): remove this hack
            if cls_name == "All2AllSoftmax":
                cls_name = "All2All"
            to_print[cls_name] = self.forward[i]
            yaml.dump(to_print, stream)
        stream.close()

    def _save_numpy_to_file(self, numpy_vector, numpy_vector_name, unit_number,
                            path):
        """Save numpy array to binary file.
        Parameters:
            numpy_vector: contains numpy array.
            numpy_vector_name: name of the numpy array.
            unit_number: number of unit that contains this numpy array.
            path: path to folder to save binary file.
        """
        link_to_numpy = "unit" + str(unit_number) + numpy_vector_name + ".bin"

        setattr(self.forward[unit_number], 'link_to_' + numpy_vector_name,
                link_to_numpy)

        array_to_save = numpy.float32(numpy_vector.ravel())

        if numpy_vector_name == "weights":
            self.log().debug("%s\n1: %f 2: %f" % (link_to_numpy,
                             numpy_vector[0][0], numpy_vector[0][1]))
        f = open("%s/%s" % (path, link_to_numpy), "wb")
        f.write(array_to_save)
        f.close()
        return link_to_numpy
