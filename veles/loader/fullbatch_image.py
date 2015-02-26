# -*- coding: utf-8 -*-
"""
Created on Feb 2, 2015

Ontology of image loading classes (full batch branch).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import numpy
from psutil import virtual_memory
from zope.interface import implementer

from veles import error
from veles.external.progressbar import ProgressBar, Percentage, Bar
from veles.loader.fullbatch import IFullBatchLoader, FullBatchLoader, \
    FullBatchLoaderMSEMixin
from veles.loader.image import ImageLoader
from veles.loader.file_image import IFileImageLoader, \
    FileListImageLoader, AutoLabelFileImageLoader, FileImageLoader
from veles.loader.image_mse import ImageLoaderMSEMixin, FileImageLoaderMSEMixin
from veles.memory import Vector


@implementer(IFullBatchLoader)
class FullBatchImageLoader(ImageLoader, FullBatchLoader):
    """Loads all images into the memory.
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchImageLoader, self).__init__(workflow, **kwargs)
        self.original_label_values = Vector()

    class DistortionIterator(object):
        def __init__(self, data, loader):
            self.data = data
            self.loader = loader
            stages = []
            for rot in self.loader.rotations:
                mirror_state = False
                if self.loader.mirror == "random":
                    mirror_state = bool(self.loader.prng.randint(2))
                stages.append((mirror_state, rot))
                if self.loader.mirror is True:
                    stages.append((True, rot))
            self.state = iter(stages)

        def __next__(self):
            return self.loader.distort(self.data, *next(self.state))

        def __iter__(self):
            return self

    def load_data(self):
        super(FullBatchImageLoader, self).load_data()

        # Allocate data
        required_mem = self.total_samples * numpy.prod(self.shape) * \
            numpy.dtype(self.source_dtype).itemsize
        if virtual_memory().available < required_mem:
            gb = 1.0 / (1000 * 1000 * 1000)
            self.critical("Not enough memory (free %.3f Gb, required %.3f Gb)",
                          virtual_memory().free * gb, required_mem * gb)
            raise MemoryError("Not enough memory")
        # Real allocation will still happen during the second pass
        self.create_originals(self.shape)
        self.original_label_values.mem = numpy.zeros(
            self.total_samples, numpy.float32)

        has_labels = self._fill_original_data()

        # Delete labels mem if no labels was extracted
        if numpy.prod(has_labels) == 0 and sum(has_labels) > 0:
            raise error.BadFormatError(
                "Some classes do not have labels while other do")
        if sum(has_labels) == 0:
            del self.original_labels[:]

    def initialize(self, device, **kwargs):
        """
        This method MUST exist to fix the diamond inherited signature.
        """
        super(FullBatchImageLoader, self).initialize(device=device, **kwargs)

    def _resize_validation(self):
        if self.validation_ratio is None:
            return
        self.debug("Skipped _resize_validation()")

    def _load_distorted_keys(self, keys, data, labels, label_values, offset,
                             pbar):
        has_labels = False
        for key in keys:
            img, _, bbox = self._load_image(key, crop=False)
            label, has_labels = self._load_label(key, has_labels)
            for ci in range(self.crop_number):
                if self.crop is not None:
                    cropped, label_value = self.crop_image(img, bbox)
                else:
                    cropped = img
                    label_value = 1.0
                for dist in FullBatchImageLoader.DistortionIterator(
                        cropped, self):
                    data[offset] = self.distort(cropped, *dist)
                    labels[offset] = label
                    label_values[offset] = label_value
                    offset += 1
                    if pbar is not None:
                        pbar.inc()
        return offset, has_labels

    def fill_minibatch(self):
        FullBatchLoader.fill_minibatch(self)
        if self.epoch_ended and self.crop is not None:
            # Overwrite original_data
            self.original_data.map_invalidate()
            self._fill_original_data()

    def _fill_original_data(self):
        pbar = ProgressBar(
            term_width=50, maxval=self.total_samples * self.samples_inflation,
            widgets=["Loading %dx%d images " % (self.total_samples,
                                                self.crop_number),
                     Bar(), ' ', Percentage()],
            log_level=logging.INFO, poll=0.5)
        pbar.start()
        offset = 0
        has_labels = []
        data = self.original_data.mem
        label_values = self.original_label_values.mem
        for keys in self.class_keys:
            if len(keys) == 0:
                continue
            if self.samples_inflation == 1:
                labels = [None] * len(keys)
                has_labels.append(self.load_keys(
                    keys, pbar, data[offset:], labels,
                    label_values[offset:]))
                offset += len(keys)
            else:
                labels = [None] * len(keys) * self.samples_inflation
                offset, hl = self._load_distorted_keys(
                    keys, data, labels, label_values, offset, pbar)
                has_labels.append(hl)
            self.original_labels[offset - len(labels):offset] = labels
        pbar.finish()
        return has_labels


class FullBatchImageLoaderMSEMixin(ImageLoaderMSEMixin,
                                   FullBatchLoaderMSEMixin):
    """
    FullBatchImageLoaderMSE implementation for parallel inheritance.
    """

    def load_data(self):
        super(FullBatchImageLoaderMSEMixin, self).load_data()
        length = len(self.target_keys) * self.samples_inflation
        targets = numpy.zeros(
            (length,) + self.targets_shape, dtype=self.source_dtype)
        target_labels = [None] * length
        has_labels = self.load_target_keys(
            self.target_keys, targets, target_labels)
        if not has_labels:
            if self.has_labels:
                raise error.BadFormatError(
                    "Targets do not have labels, but the classes do")
            # Associate targets with classes by sequence order
            self.original_targets.mem = targets
            return
        if not self.has_labels:
            raise error.BadFormatError(
                "Targets have labels, but the classes do not")
        if len(set(target_labels)) < length / self.samples_inflation:
            raise error.BadFormatError("Some targets have duplicate labels")
        diff = set(self.original_labels).difference(target_labels)
        if len(diff) > 0:
            raise error.BadFormatError(
                "Labels %s do not have corresponding targets" % diff)
        self.original_targets.reset()
        shape = (len(targets),) + targets[0].shape
        self.original_targets.mem = numpy.zeros(
            (len(self.original_labels),) + targets[0].shape,
            self.source_dtype)
        target_mapping = {
            target_labels[i * self.samples_inflation]: targets[i]
            for i in range(length // self.samples_inflation)}
        self.class_targets.reset(numpy.empty(shape, self.source_dtype))
        for i, label in enumerate(target_labels):
            self.class_targets[i] = target_mapping[label]
        for i, label in enumerate(self.original_labels):
            self.original_targets[i] = target_mapping[label]


class FullBatchImageLoaderMSE(FullBatchImageLoaderMSEMixin,
                              FullBatchImageLoader):
    """
    MSE modification of FullBatchImageLoader class.
    """
    pass


@implementer(IFileImageLoader)
class FullBatchFileListImageLoader(FileListImageLoader, FullBatchImageLoader):
    MAPPING = "full_batch_file_list_image"


class FullBatchAutoLabelFileImageLoader(AutoLabelFileImageLoader,
                                        FullBatchImageLoader):
    """
    Full batch variant of AutoLabelFileImageLoader.
    """
    MAPPING = "full_batch_auto_label_file_image"


class FullBatchFileImageLoader(FileImageLoader, FullBatchImageLoader):
    """Loads images from multiple folders as full batch.
    """
    pass


class FullBatchFileImageLoaderMSEMixin(FullBatchImageLoaderMSEMixin,
                                       FileImageLoaderMSEMixin):
    pass


class FullBatchFileImageLoaderMSE(FullBatchFileImageLoaderMSEMixin,
                                  FullBatchFileImageLoader):
    """
    MSE modification of  FullBatchFileImageLoader class.
    """
    pass


@implementer(IFileImageLoader)
class FullBatchAutoLabelFileImageLoaderMSE(
        AutoLabelFileImageLoader, FullBatchFileImageLoaderMSE):
    MAPPING = "full_batch_auto_label_file_image_mse"
