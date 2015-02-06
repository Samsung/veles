# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2015

Ontology of image loading classes.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from itertools import chain
import numpy

from veles import error
from veles.loader.base import LoaderMSEMixin, TARGET, Loader
from veles.loader.image import ImageLoader, FileImageLoader


class ImageLoaderMSEMixin(LoaderMSEMixin):
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
        if self._restored_from_pickle:
            return
        self.target_keys.extend(self.get_keys(TARGET))
        length = len(self.target_keys)
        if len(set(self.target_keys)) < length:
            raise error.BadFormatError("Some targets have duplicate keys")
        self.target_keys.sort()
        if not self.has_labels and length != sum(self.class_lengths):
            raise error.BadFormatError(
                "Number of class samples %d differs from the number of "
                "targets %d" % (sum(self.class_lengths), length))
        if self.has_labels:
            labels = numpy.zeros(length, dtype=Loader.LABEL_DTYPE)
            assert self.load_keys(self.target_keys, None, None, labels, None)
            if len(set(labels)) < length:
                raise error.BadFormatError("Targets have duplicate labels")
            self.target_label_map = {l: self.target_keys[l] for l in labels}

    def create_minibatches(self):
        super(ImageLoaderMSEMixin, self).create_minibatches()
        self.minibatch_targets.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype))

    def fill_minibatch(self):
        super(ImageLoaderMSEMixin, self).fill_minibatch()
        indices = self.minibatch_indices.mem[:self.minibatch_size]
        if not self.has_labels:
            keys = self.keys_from_indices(self.shuffled_indices[i]
                                          for i in indices)
        else:
            keys = (self.target_label_map[l]
                    for l in self.minibatch_labels.mem)
        assert self.has_labels == self.load_keys(
            keys, None, self.minibatch_targets.mem, None, None)


class ImageLoaderMSE(ImageLoaderMSEMixin, ImageLoader):
    """
    Loads images in MSE schemes. Like ImageLoader, mostly useful for large
    datasets.
    """
    pass


class FileImageLoaderMSEMixin(ImageLoaderMSEMixin):
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
