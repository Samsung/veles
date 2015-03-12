"""
Created on Jan 25, 2015

Loaders which get data from pickles

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import pickle

import numpy
import six
from zope.interface import implementer

from veles import error
from veles.compat import from_none
from veles.external.progressbar import ProgressBar
from veles.memory import interleave
from veles.loader.base import CLASS_NAME, Loader
from veles.loader.image import IImageLoader, COLOR_CHANNELS_MAP
from veles.loader.fullbatch import FullBatchLoader, IFullBatchLoader
from veles.loader.fullbatch_image import FullBatchImageLoader


@implementer(IFullBatchLoader)
class PicklesLoader(FullBatchLoader):
    """
    Loads samples from pickles for data set.
    """
    def __init__(self, workflow, **kwargs):
        super(PicklesLoader, self).__init__(workflow, **kwargs)
        self._test_pickles = list(kwargs.get("test_pickles", []))
        self._validation_pickles = list(kwargs.get("validation_pickles", []))
        self._train_pickles = list(kwargs.get("train_pickles", []))
        self._pickles = (self.test_pickles, self.validation_pickles,
                         self.train_pickles)

    @property
    def test_pickles(self):
        return self._test_pickles

    @property
    def validation_pickles(self):
        return self._validation_pickles

    @property
    def train_pickles(self):
        return self._train_pickles

    def reshape(self, shape):
        return shape

    def transform_data(self, data):
        return data

    def load_data(self):
        pbar = ProgressBar(maxval=sum(len(p) for p in self._pickles),
                           term_width=40)
        self.info("Loading %d pickles...", pbar.maxval)
        pbar.start()
        loaded = [self.load_pickles(i, self._pickles[i], pbar)
                  for i in range(3)]
        pbar.finish()
        self.info("Initializing the arrays...")
        shape = loaded[2][1][0].shape[1:]
        for i in range(2):
            if loaded[i][0] > 0:
                shi = loaded[i][1][0].shape[1:]
                if shape != shi:
                    raise error.BadFormatError(
                        "TRAIN and %s sets have the different sample shape "
                        "(%s vs %s)" % (CLASS_NAME[i], shape, shi))
        self.create_originals(self.reshape(shape))
        offsets = [0, 0]
        for ds in range(3):
            if loaded[ds][0] == 0:
                continue
            for arr in loaded[ds][1]:
                self.original_data[offsets[0]:(offsets[0] + arr.shape[0])] = \
                    self.transform_data(arr)
                offsets[0] += arr.shape[0]
            for arr in loaded[ds][2]:
                self.original_labels[offsets[1]:(offsets[1] + arr.shape[0])] =\
                    arr
                offsets[1] += arr.shape[0]

    def load_pickles(self, index, pickles, pbar):
        unpickled = []
        for pick in pickles:
            try:
                with open(pick, "rb") as fin:
                    self.debug("Loading %s...", pick)
                    if six.PY3:
                        loaded = pickle.load(fin, encoding='charmap')
                    else:
                        loaded = pickle.load(fin)
                    unpickled.append(loaded)
                    pbar.inc()
            except Exception as e:
                self.warning(
                    "Failed to load %s (part of %s set)" %
                    (pick, CLASS_NAME[index]))
                raise from_none(e)
        data = []
        labels = []
        for obj, pick in zip(unpickled, pickles):
            if not isinstance(obj, dict):
                raise TypeError(
                    "%s has the wrong format (part of %s set)" %
                    (pick, CLASS_NAME[index]))
            try:
                data.append(obj["data"])
                labels.append(
                    numpy.array(obj["labels"], dtype=Loader.LABEL_DTYPE))
            except KeyError as e:
                self.error("%s has the wrong format (part of %s set)",
                           pick, CLASS_NAME[index])
                raise from_none(e)
        lengths = [0, sum(len(l) for l in labels)]
        for arr in data:
            lengths[0] += arr.shape[0]
            if arr.shape[1:] != data[0].shape[1:]:
                raise error.BadFormatError(
                    "Array has a different shape: expected %s, got %s"
                    "(%s set)" % (data[0].shape[1:],
                                  arr.shape[1:], CLASS_NAME[index]))
        if lengths[0] != lengths[1]:
            raise error.BadFormatError(
                "Data and labels has the different number of samples (data %d,"
                " labels %d)" % lengths)
        length = lengths[0]
        self.class_lengths[index] = length
        return length, data, labels


@implementer(IImageLoader)
class PicklesImageFullBatchLoader(PicklesLoader, FullBatchImageLoader):
    MAPPING = "full_batch_pickles_image"

    def __init__(self, workflow, **kwargs):
        super(PicklesImageFullBatchLoader, self).__init__(workflow, **kwargs)
        # Since we can not extract the color space information from pickles
        # set it explicitly without any default value
        self.color_space = kwargs["color_space"]

    def get_image_label(self, key):
        return int(self.image_labels[key])

    def get_image_info(self, key):
        return self.image_data[key].shape[:2], self.color_space

    def get_image_data(self, key):
        return self.image_data[key]

    def get_keys(self, index):
        offsets = [0, self.class_lengths[0],
                   self.class_lengths[0] + self.class_lengths[1],
                   self.total_samples]
        self.original_shape = self.image_data.shape[1:-1]
        return range(offsets[index], offsets[index + 1])

    def reshape(self, shape):
        if shape[0] == COLOR_CHANNELS_MAP[self.color_space]:
            return shape[1:] + (shape[0],)
        return shape

    def transform_data(self, data):
        if data.shape[1] == COLOR_CHANNELS_MAP[self.color_space]:
            return interleave(data)
        return data

    def load_data(self):
        PicklesLoader.load_data(self)
        self.original_class_lengths = self.class_lengths
        self.image_data = self.original_data.mem
        self.original_data.mem = None
        self.image_labels = self.original_labels[:]
        del self.original_labels[:]
        FullBatchImageLoader.load_data(self)
        assert self.original_class_lengths == self.class_lengths
        del self.image_data

    def initialize(self, device, **kwargs):
        super(PicklesImageFullBatchLoader, self).initialize(
            device=device, **kwargs)
        del self.image_labels
