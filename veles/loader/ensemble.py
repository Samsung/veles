# -*- coding: utf-8 -*-  # pylint: disable=C0302
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 17, 2015

Loader for the ensemble_train (--ensemble_train) output.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import json
import numpy
from zope.interface import implementer, Interface

from .base import TEST, VALID, TRAIN
from .fullbatch import FullBatchLoader, IFullBatchLoader, FullBatchLoaderMSE


class IEnsembleLoader(Interface):
    def load_winners():
        """
        :return: tuple (list of winning labels, <labels are indices>)
        """


class EnsembleLoaderBase(FullBatchLoader):
    def __init__(self, workflow, **kwargs):
        super(EnsembleLoaderBase, self).__init__(workflow, **kwargs)
        self._file = kwargs["file"]

    @property
    def file(self):
        return self._file

    def _load_data_from_file(self):
        with open(self.file, "r") as fin:
            return json.load(fin)

    def _load_outputs(self, data):
        outputs = []
        for model in data["models"]:
            mid = model["id"]
            outputs.append(numpy.array(model["Output"]))
            if outputs[-1].shape != outputs[0].shape:
                raise ValueError(
                    "Model with id %s has an invalid output shape %s vs %s "
                    "mapping" % (mid, outputs[-1].shape, outputs[0].shape))
        return outputs

    def _fill_class_lengths(self, outputs):
        if not self.testing:
            self.class_lengths[TEST] = self.class_lengths[VALID] = 0
            self.class_lengths[TRAIN] = len(outputs[0])
        else:
            self.class_lengths[TRAIN] = self.class_lengths[VALID] = 0
            self.class_lengths[TEST] = len(outputs[0])

    def _fill_original_data(self, outputs, labels):
        self.create_originals((len(outputs),) + outputs[0].shape[1:],
                              not self.testing and labels)
        for oi, output in enumerate(outputs):
            for i, v in enumerate(output):
                self.original_data[i, oi] = v


@implementer(IFullBatchLoader)
class EnsembleLoader(EnsembleLoaderBase):
    def __init__(self, workflow, **kwargs):
        super(EnsembleLoader, self).__init__(workflow, **kwargs)
        if not self.testing:
            self.verify_interface(IEnsembleLoader)

    def load_data(self):
        data = self._load_data_from_file()
        outputs = self._load_outputs(data)
        labels_mapping = None
        reversed_labels_mapping = None
        for mi, model in enumerate(data["models"]):
            mid = model["id"]
            labels = model["Labels"]
            if reversed_labels_mapping is None:
                reversed_labels_mapping = labels
                labels_mapping = {
                    v: i for i, v in enumerate(reversed_labels_mapping)}
            elif reversed_labels_mapping != labels:
                if len(reversed_labels_mapping) != len(labels):
                    raise ValueError(
                        "Model with id %s has completely different labels "
                        "mapping" % mid)
                self.warning("Model with id %s has a different labels mapping,"
                             " remapping", mid)
                # remap labels
                output = numpy.zeros_like(outputs[mi])
                for si, (src, dst) in enumerate(zip(outputs[mi], output)):
                    for i, v in enumerate(src):
                        dst[labels_mapping[labels[i]]] = v
        self._fill_class_lengths(outputs)
        self._fill_original_data(outputs, True)
        if not self.testing:
            true_labels, format_indices = self.load_winners()
            if format_indices:
                self.original_labels[:] = true_labels
            else:
                self.original_labels[:] = (
                    labels_mapping[l] for l in true_labels)


class IEnsembleLoaderMSE(Interface):
    def load_targets():
        """
        Fill original_targets here.
        """


@implementer(IFullBatchLoader)
class EnsembleLoaderMSE(EnsembleLoaderBase, FullBatchLoaderMSE):
    def __init__(self, workflow, **kwargs):
        super(EnsembleLoaderMSE, self).__init__(workflow, **kwargs)
        self.verify_interface(IEnsembleLoaderMSE)

    def load_data(self):
        data = self._load_data_from_file()
        outputs = self._load_outputs(data)
        self._fill_class_lengths(outputs)
        self._fill_original_data(outputs, False)
        self.load_targets()
        yours_shape = self.original_targets.shape
        mine_shape = (len(self.original_data),) + outputs[0].shape[1:]
        if yours_shape != mine_shape:
            raise ValueError(
                "Invalid original_targets shape %s vs %s" %
                (yours_shape, mine_shape))
