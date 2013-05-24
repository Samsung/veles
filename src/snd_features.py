"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from decimal import Decimal
from itertools import groupby
import logging
import numpy
from xml.etree import ElementTree
import units
from sound_feature_extraction.extractor import Extractor
from sound_feature_extraction.feature import Feature
from inspect import isclass


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
        description = description.strip()
        logging.debug("Adding \"" + description + "\"")
        feature = Feature.from_string(description)
        self.features.append(feature)

    def add_features(self, descriptions):
        for desc in descriptions:
            self.add_feature(desc)

    def initialize(self):
        # Validate the set features by constructing an Extractor instance
        Extractor(self.features, 1000, 16000)

    def run(self):
        inputs = {}
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
                        self.outputs.append(extr.calculate(data["data"]))
                    except Exception as e:
                        logging.warn("Failed to extract features from "
                                     "input: " + repr(e))
                        self.outputs.append(None)

    def save_to_file(self, file_name, labels):
        if len(labels) != len(self.outputs):
            logging.error("Labels and outputs size mismatch")
        root = ElementTree.Element("features", {"version": "1.0"})
        for i in range(0, len(labels)):
            label = labels[i]
            file_element = ElementTree.SubElement(root,
                                                  "file", {"name": label})
            for feature in self.features:
                feat_element = ElementTree.SubElement(
                    file_element, "feature",
                    {"description": feature.description(),
                     "name": feature.name})
                if self.outputs[i]:
                    if len(self.outputs[i][feature.name].shape) > 0:
                        feat_element.attrib["value"] = "".join(\
                            str(Decimal(el).normalize()) + " " \
                            for el in self.outputs[i][feature.name])
                    else:
                        feat_element.attrib["value"] = \
                            str(self.outputs[i][feature.name])
        ElementTree.ElementTree(root).write(file_name, encoding="utf-8",
                                            xml_declaration=True)
