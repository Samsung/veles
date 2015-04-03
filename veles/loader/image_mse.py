# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Feb 4, 2015

Ontology of image loading classes

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


from itertools import chain
import numpy

from veles import error
from veles.loader.base import LoaderMSEMixin, TARGET
from veles.loader.image import ImageLoader
from veles.loader.file_image import FileImageLoader


class ImageLoaderMSEMixin(LoaderMSEMixin):
    hide_from_registry = True
    """
    Implementation of ImageLoaderMSE for parallel inheritance.

    Attributes:
        target_keys: additional key list of targets.
    """

    def __init__(self, workflow, **kwargs):
        super(ImageLoaderMSEMixin, self).__init__(workflow, **kwargs)
        self.target_keys = []
        self.target_label_map = None

    def load_data(self):
        super(ImageLoaderMSEMixin, self).load_data()
        if self._restored_from_pickle_:
            return
        if len(self.target_keys) == 0:
            self.target_keys.extend(self.get_keys(TARGET))
        length = len(self.target_keys)
        if len(set(self.target_keys)) < length:
            raise error.BadFormatError("Some targets have duplicate keys")
        self.target_keys.sort()
        if not self.has_labels and length != self.total_samples:
            raise error.BadFormatError(
                "Number of class samples %d differs from the number of "
                "targets %d" % (self.total_samples, length))
        if self.has_labels:
            labels = [None] * length
            assert self.load_target_keys(self.target_keys, None, labels)
            if len(set(labels)) < length:
                raise error.BadFormatError("Targets have duplicate labels")
            self.target_label_map = {
                l: k for l, k in zip(labels, self.target_keys)}

    def load_target_keys(self, keys, data, labels):
        """Loads data from the specified keys.
        """
        index = 0
        has_labels = False
        for key in keys:
            obj = self.load_target(key)
            label, has_labels = self._load_label(key, has_labels)
            if data is not None:
                data[index] = obj
            if labels is not None:
                labels[index] = label
            index += 1
        return has_labels

    def load_target(self, key):
        return self.get_image_data(key)

    def create_minibatch_data(self):
        super(ImageLoaderMSEMixin, self).create_minibatch_data()
        self.minibatch_targets.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.targets_shape, dtype=self.dtype))

    def fill_minibatch(self):
        super(ImageLoaderMSEMixin, self).fill_minibatch()
        indices = self.minibatch_indices.mem[:self.minibatch_size]
        if not self.has_labels:
            keys = self.keys_from_indices(self.shuffled_indices[i]
                                          for i in indices)
        else:
            keys = (self.target_label_map[l]
                    for l in self.minibatch_labels.mem)
        assert self.has_labels == self.load_target_keys(
            keys, self.minibatch_targets.mem, None)


class ImageLoaderMSE(ImageLoaderMSEMixin, ImageLoader):
    """
    Loads images in MSE schemes. Like ImageLoader, mostly useful for large
    datasets.
    """
    pass


class FileImageLoaderMSEMixin(ImageLoaderMSEMixin):
    hide_from_registry = True
    """
    FileImageLoaderMSE implementation for parallel inheritance.

    Attributes:
        target_paths: list of paths for target in case of MSE.
    """

    def __init__(self, workflow, **kwargs):
        super(FileImageLoaderMSEMixin, self).__init__(
            workflow, **kwargs)
        self.target_paths = kwargs["target_paths"]

    @property
    def target_paths(self):
        return self._target_paths

    @target_paths.setter
    def target_paths(self, value):
        self._check_paths(value)
        self._target_paths = value

    def get_keys(self, index):
        if index != TARGET:
            return super(FileImageLoaderMSEMixin, self).get_keys(
                index)
        return list(chain.from_iterable(
            self.scan_files(p) for p in self.target_paths))


class FileImageLoaderMSE(FileImageLoaderMSEMixin, FileImageLoader):
    """
    MSE modification of  FileImageLoader class.
    """
    pass
