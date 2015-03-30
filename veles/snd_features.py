"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from itertools import groupby
import logging
import os

from libSoundFeatureExtraction.python.sound_feature_extraction import extractor
from libSoundFeatureExtraction.python.sound_feature_extraction import feature
import veles.units as units
from veles.pickle2 import pickle, best_protocol


class SoundFeatures(units.Unit):
    """
    Extracts features from raw audio data.
    """

    def __init__(self, report_path, workflow, name=None):
        super(SoundFeatures, self).__init__(workflow=workflow, name=name,
                                            view_group="WORKER")
        self.features = []
        self.inputs = []
        self.outputs = []
        self.report_path = report_path

    def add_feature(self, description):
        description = description.strip()
        logging.debug("Adding \"" + description + "\"")
        self.features.append(feature.Feature.from_string(description))

    def add_features(self, descriptions):
        for desc in descriptions:
            self.add_feature(desc)

    def initialize(self, **kwargs):
        pass

    def extract(self, name, data, extr):
        try:
            logging.info("Extracting features from " + name)
            result = extr.calculate(data)
            if self.report_path is not None:
                extr.report(os.path.join(self.report_path,
                                         os.path.basename(name) + ".dot"))
            return result
        except Exception as e:
            logging.warn("Failed to extract features from input: " + repr(e))
            return None

    def run(self):
        sorted_inputs = {}
        sorted_outputs = {}
        self.outputs = []
        for channels, grch in groupby(
                sorted(self.inputs, key=lambda x: x["channels"]),
                lambda x: x["channels"]):
            sorted_inputs[channels] = {}
            for sampling_rate, grsr in groupby(
                    sorted(grch, key=lambda x: x["sampling_rate"]),
                    lambda x: x["sampling_rate"]):
                sorted_inputs[channels][sampling_rate] = {}
                for size, grsz in groupby(
                        sorted(grsr, key=lambda x: x["data"].size),
                        lambda x: x["data"].size):
                    sorted_inputs[channels][sampling_rate][size] = list(grsz)
        for channels, grch in sorted_inputs.items():
            for sampling_rate, grsr in grch.items():
                for size, grsz in grsr.items():
                    extr = extractor.Extractor(self.features, size,
                                               sampling_rate, channels)
                    for data in sorted_inputs[channels][sampling_rate][size]:
                        sorted_outputs[data["name"]] = (
                            self.extract(data["name"], data["data"], extr),
                            sampling_rate, channels)
        # Fill self.outputs from sorted_outputs in self.inputs order
        for inp in self.inputs:
            self.outputs.append(sorted_outputs[inp["name"]])

    def save_to_file(self, file_name, labels):
        if len(labels) != len(self.outputs):
            raise Exception("Labels and outputs size mismatch (" +
                            str(len(labels)) + " vs " +
                            str(len(self.outputs)) + ")")
        logging.debug("Saving %d results", len(labels))
        root = {"version": "1.0", "files": {}}
        indices_map = sorted(range(0, len(labels)), key=lambda x: labels[x])
        labels.sort()
        for j in range(0, len(labels)):
            i = indices_map[j]
            label = labels[j]
            file_element = {"features": {}}
            for features in self.features:
                feat_element = {"description": features.description(
                    {"sampling_rate": self.outputs[i][1],
                     "channels": self.outputs[i][2]})}
                if self.outputs[i]:
                    feat_element["value"] = self.outputs[i][0][features.name]
                file_element["features"][features.name] = feat_element
            root["files"][label] = file_element
        fout = open(file_name, "wb")
        pickle.dump(root, fout, protocol=best_protocol)
