"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from decimal import Decimal
from itertools import groupby
import logging
import pickle
import units
from sound_feature_extraction.extractor import Extractor
from sound_feature_extraction.feature import Feature


class SoundFeatures(units.Unit):
    """
    Extracts features from raw audio data.
    """

    def __init__(self):
        super(SoundFeatures, self).__init__()
        self.features = []
        self.inputs = []
        self.outputs = []

    def add_feature(self, description):
        description = description.strip()
        logging.debug("Adding \"" + description + "\"")
        feature = Feature.from_string(description)
        self.features.append(feature)

    def add_features(self, descriptions):
        for desc in descriptions:
            self.add_feature(desc)

    def initialize(self):
        pass

    def run(self):
        inputs = {}
        self.outputs = []
        for sampling_rate, grsr in groupby(
            sorted(self.inputs, key=lambda x: x["sampling_rate"]),
            lambda x: x["sampling_rate"]):
            inputs[sampling_rate] = {}
            for size, grsz in groupby(
                sorted(grsr, key=lambda x: x["data"].size),
                lambda x: x["data"].size):
                inputs[sampling_rate][size] = list(grsz)
        for sampling_rate in inputs:
            for size in inputs[sampling_rate]:
                extr = Extractor(self.features, size, sampling_rate)
                for data in inputs[sampling_rate][size]:
                    try:
                        logging.info("Extracting features from " +
                                      data["name"])
                        self.outputs.append(extr.calculate(data["data"]))
                    except Exception as e:
                        logging.warn("Failed to extract features from "
                                     "input: " + repr(e))
                        self.outputs.append(None)

    def save_to_file(self, file_name, labels):
        if len(labels) != len(self.outputs):
            raise Exception("Labels and outputs size mismatch (" +
                            str(len(labels)) + " vs " +
                            str(len(self.outputs)) + ")")
        root = {"version": "1.0", "files": {}}
        indices_map = sorted(range(0, len(labels)), key=lambda x: labels[x])
        labels.sort()
        for j in range(0, len(labels)):
            i = indices_map[j]
            label = labels[j]
            file_element = {"features": {}}
            for feature in self.features:
                feat_element = {"description": feature.description()}
                if self.outputs[i]:
                    feat_element["value"] = self.outputs[i][feature.name]
                file_element["features"][feature.name] = feat_element
            root["files"][label] = file_element
        fout = open(file_name, "wb")
        pickle.dump(root, fout)
