"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jan 27, 2015

Base classes to load data from HDF5 (Caffe format).

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


from collections import Counter
import h5py
import numpy
from zope.interface import implementer

from veles import error
from veles.loader.base import ILoader, Loader
from veles.loader.fullbatch import IFullBatchLoader, FullBatchLoader


class HDF5LoaderBase(Loader):
    def __init__(self, workflow, **kwargs):
        super(HDF5LoaderBase, self).__init__(workflow, **kwargs)
        self._files = (kwargs.get("test_path"),
                       kwargs.get("validation_path"),
                       kwargs.get("train_path"))
        self._shape = None

    @property
    def files(self):
        return self._files

    @Loader.shape.getter
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if self._shape is None:
            self._shape = value
        else:
            assert self._shape == value

    def open_hdf5(self, index):
        path = self._files[index]
        if not path:
            return None, None
        h5f = h5py.File(path)
        data = h5f["data"]
        has_labels = "label" in h5f
        if self.has_labels and not has_labels or \
                not self.has_labels and has_labels and \
                self.total_samples > 0:
            raise error.BadFormatError(
                "Some sets have labels and some do not")
        self._has_labels = has_labels
        labels = h5f["label"] if self.has_labels else None
        if self.has_labels and len(data) != len(labels):
            raise error.BadFormatError(
                "%s: data and labels have different lengths" % path)
        self.class_lengths[index] = len(data)
        self.shape = data.shape[1:]
        return data, labels


@implementer(ILoader)
class HDF5Loader(HDF5LoaderBase):
    def __init__(self, workflow, **kwargs):
        super(HDF5Loader, self).__init__(workflow, **kwargs)
        self._datasets = [None] * 3

    def load_data(self):
        for index in range(3):
            self._datasets[index] = self.open_hdf5(index)
        diff_labels = tuple(Counter(self._datasets[i][1]) for i in range(3))
        self._setup_labels_mapping(diff_labels)

    def create_minibatch_data(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        self.minibatch_data.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype))

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        for i, sample_index in enumerate(
                self.minibatch_indices.mem[:self.minibatch_size]):
            ci, rem = self.class_index_by_sample_index(sample_index)
            dataset = self._datasets[ci]
            offset = self.class_lengths[ci] - rem
            self.minibatch_data[i] = dataset[0][offset]
            if self.has_labels:
                self.raw_minibatch_labels[i] = dataset[1][offset]


@implementer(IFullBatchLoader)
class FullBatchHDF5Loader(FullBatchLoader, HDF5LoaderBase):
    @FullBatchLoader.shape.getter
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if self._shape is None:
            self._shape = value
        else:
            assert self._shape == value

    def load_data(self):
        all_data = [None] * 3
        all_labels = [None] * 3
        for index in range(3):
            all_data[index], all_labels[index] = self.open_hdf5(index)
        self.create_originals(self.shape)
        offset = 0
        for data, labels, length in zip(all_data, all_labels,
                                        self.class_lengths):
            if data is None or length == 0:
                continue
            next_offset = offset + length
            self.original_data[offset:next_offset] = data[:]
            self.original_labels[offset:next_offset] = labels[:]
            offset = next_offset
