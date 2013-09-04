"""

Saver unit.

@author: EBulychev
"""
import os
import yaml
import formats
import units
import logging as log
import tarfile
import numpy as np
import shutil

# @brief Saver unit
class SaverUnit(units.Unit):
    """SaverUnit class purpose:  save All2All.Forward units to archive

    Should be assigned before initialize():        

    Updates after run(): create archive that contains yaml-file with workflow
    attributes + binary files with numpy-arrays.

    Creates within initialize():
        stream to file 

    Attributes:
        forward: list of units that inherits from Forward.
    """
    def __init__(self, forward_list, filename, unpickling=0):
        super(SaverUnit, self).__init__()
        self.forward = forward_list
        self.filename = filename


    def initialize(self):
        pass
    # @brief Check that object is class.
    # @param[in] Obj_to_check Object that should be checked.
    # @return Return True if object is class.
    def isClassInsideObject(self, Obj_to_check):
        if isinstance(Obj_to_check, (str, bytes, bool, int, float,
                                     list, tuple)) == False :
            return True
        return False

    # @brief Print workflow to yaml-file
    # @param[in] yaml_name Filename to save.
    def save_to_yaml(self, yaml_name):
        stream = open(yaml_name, 'w')

        for i in range(len(self.forward)):
            toPrint = {}
            cls = self.forward[i].__class__
            log.debug("Saving " + cls.__name__ + "...")
            cls.yaml_dumper.add_representer(cls, cls.to_yaml)
            cls_name = self.forward[i].__class__.__name__
            # TODO(EBulychev): remove this hack
            if cls_name == "All2AllSoftmax":
                cls_name = "All2All"
            toPrint[cls_name] = self.forward[i]
            yaml.dump(toPrint, stream)
        stream.close()
        pass

    # @brief Save numpy array to binary file.
    # @param[in] numpy_vector This variable contains numpy array.
    # @param[in] numpy_vector_name Name of numpy array.
    # @param[in] unit_number Count of unit that contains this numpy array.
    # @param[in] path Path to folder to save binary file.
    def saveNumpyToFile(self, numpy_vector, numpy_vector_name, unit_number,
                        path):
        link_to_numpy = "unit" + str(unit_number) + numpy_vector_name + ".bin"

        setattr(self.forward[unit_number], 'link_to_' + numpy_vector_name,
                link_to_numpy)

        array_to_save = np.float32(numpy_vector.ravel())

        if numpy_vector_name == "weights":
            log.debug("%s\n1: %f 2: %f" % (link_to_numpy, numpy_vector[0][0], numpy_vector[0][1]))
        elif numpy_vector_name == "output" and unit_number == 1:
            for i in range(numpy_vector.shape[0]):
                for j in range(numpy_vector.shape[1]):
                    log.debug("(%d,%d) %f" % (i, j, numpy_vector[i][j]))
                log.debug("\n")
        f = open(path + link_to_numpy, "wb")
        f.write(array_to_save)
        f.close()
        return link_to_numpy
        pass

    # @brief Main function.
    def run(self):
        # create temporary folder
        newpath = "/tmp/saver_tmp/"
        if not os.path.exists(newpath): os.makedirs(newpath)
        files_to_save = []
        variables_to_save = []
        # Go through forward elements & save numpy array to binary file &
        # delete some variables
        for i in range(len(self.forward)):
            variables = self.forward[i].__getstate__()
            for key in variables:
                pass_next = False
                # Save numpy array to binary file
                if (type(getattr(self.forward[i], key)) == formats.Vector) :

                    files_to_save.append(self.
                                         saveNumpyToFile(getattr(self.forward[i], key).v, key,
                                         i, newpath))
                    pass_next = True
                    delattr(self.forward[i], key)
                # Delete links_to & links_from variables.
                if (variables[key] == None or
                    key == 'links_to' or
                    key == 'links_from' and pass_next == False):
                    delattr(self.forward[i], key)
                    pass_next = True
                    pass
                # Delete complex variables that can not been save to yaml.
                if pass_next == False and self.isClassInsideObject(getattr(self.forward[i], key)):
                    delattr(self.forward[i], key)
                    pass_next = True
            variables = self.forward[i].__getstate__()
            variables_to_save.append(variables)

        # Save forward elements to yaml.
        yaml_name = 'default.yaml'
        self.save_to_yaml(newpath + yaml_name)
        # Compress archive
        tar = tarfile.open(self.filename + ".tar.gz", "w:gz")
        tar.add(newpath + yaml_name,
                arcname=yaml_name, recursive=False)
        for i in range(len(files_to_save)):
            tar.add(newpath + files_to_save[i],
                    arcname=files_to_save[i], recursive=False)
        tar.close()
#         # delete temporary folder
        shutil.rmtree(newpath)
