"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import logging
import numpy
import Znicz.units as units
from sound_feature_extraction.extractor import Extractor
from sound_feature_extraction.feature import Feature


class SoundFeatures(units.Unit):
    """
    Extracts features from raw audio data.
    """

    def __init__(self, unpickling=0):
        super(SoundFeatures, self).__init__(unpickling=unpickling)
        self.test_only = False
        if unpickling:
            return
        self.features = []
        self.inputs = []
        self.outputs = []

    def add_feature(self, description):
        logging.debug("Adding \"" + description + "\"")
        feature = Feature.from_string(description)
        self.features.append(feature)

    def initialize(self):
        # Validate the set features by constructing an Extractor instance
        Extractor(self.features, 1000, 16000)

    def run(self):
        for raw_data in self.inputs:
            extr = Extractor(self.features, raw_data["data"].size,
                             raw_data["sampling_rate"])
            self.outputs.append(extr.calculate(raw_data["data"]))

    def save_to_text_file(self, file_name, labels):
        if len(labels) != len(self.outputs):
            logging.error("Labels and outputs size mismatch")
        with open(file_name, "w") as file:
            for i in range(0, len(labels)):
                file.write("[%s]\n" % labels[i])
                for feature in self.features:
                    file.write(feature.name + ": ")
                    file.write("".join((str(el) + " ") \
                        for el in self.outputs[i][feature.name]))
                    file.write("\n")
