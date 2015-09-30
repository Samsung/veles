# -*- coding: utf-8 -*-  # pylint: disable=C0302
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 14, 2013

Loader base class.

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


from __future__ import division
import argparse
from collections import defaultdict
from copy import copy
import logging
import marshal
import time
import types

import numpy
from veles.cmdline import CommandLineArgumentsRegistry

try:
    from scipy.stats import chisquare
except ImportError:
    chisquare = None
import six
from zope.interface import implementer, Interface

from veles.compat import from_none, has_colors
import veles.config as config
from veles.distributable import IDistributable
import veles.error as error
from veles.external.prettytable import PrettyTable
from veles.external.progressbar import ProgressBar
import veles.memory as memory
from veles.mutable import Bool
import veles.normalization as normalization
from veles.opencl_types import dtypes
import veles.prng as random_generator
from veles.result_provider import IResultProvider
from veles.units import Unit, IUnit, nothing
from veles.unit_registry import MappedUnitRegistry

TARGET = 3
TRAIN = 2
VALID = 1
TEST = 0
TRIAGE = {"train": TRAIN,
          "validation": VALID,
          "valid": VALID,
          "test": TEST}
CLASS_NAME = ["test", "validation", "train"]


class UserLoaderRegistry(MappedUnitRegistry, CommandLineArgumentsRegistry):
    mapping = "loaders"
    base = Unit

    @staticmethod
    def get_factory(name, **kwargs):
        if name not in UserLoaderRegistry.loaders:
            raise ValueError(
                "\"%s\" is not a valid loader name. Registered names: %s." %
                (name, list(sorted(UserLoaderRegistry.loaders))))
        return lambda w: UserLoaderRegistry.loaders[name](w, **kwargs)


class LoaderError(Exception):
    pass


class ILoader(Interface):
    def load_data():
        """Initializes the instance and measures the dataset size.

        Should be filled here:
            class_lengths[].
        """

    def create_minibatch_data():
        """Allocates array for minibatch_data.
        """

    def fill_minibatch():
        """Fills minibatch data labels and indexes according to the current
        shuffle (minibatch_indices[:self.minibatch_size]).
        """


@implementer(IDistributable, IUnit, IResultProvider)
@six.add_metaclass(UserLoaderRegistry)
class Loader(Unit):
    """Loads data and provides minibatch output interface.

    Attributes:
        prng: veles.prng.RandomGenerator instance.
        max_minibatch_size: maximal size of a minibatch.
        total_samples: total number of samples in the dataset.
        class_lengths: number of samples per class.
        class_end_offsets: offset in samples where the next class begins.
        last_minibatch: if current minibatch is last in it's class.
        epoch_ended: True right after validation is completed and no samples
                     have been served since.
        epoch_number: current epoch number. Epoch ends when all validation set
                      is processed. If validation set is empty, it ends
                      after all training set is processed.
        minibatch_data: data (should be scaled usually scaled to [-1, 1]).
        minibatch_indices: global indices of images in minibatch.
        minibatch_labels: labels for indexes in minibatch (classification).
        shuffled_indices: indices for all dataset, shuffled with prng.
        samples_served: the total number of samples processed for all epochs.
        minibatch_class: current minibatch class.
        minibatch_offset: current minibatch offset.
        global_offset: first sample index which was not served during the
                       current epoch.
        minibatch_size: current minibatch size <= max_minibatch_size.
    """

    LABEL_DTYPE = numpy.int32
    INDEX_DTYPE = numpy.int32
    exports = "epoch_ended", "epoch_number", "train_ended", "class_lengths", \
        "minibatch_data", "minibatch_class", "minibatch_data", "has_labels", \
        "minibatch_labels", "minibatch_size", "max_minibatch_size", \
        "total_samples", "last_minibatch", "class_lengths", "shuffle_limit", \
        "labels_mapping", "reversed_labels_mapping", "global_offset", \
        "minibatch_offset"

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "LOADER"
        self.last_minibatch = Bool()
        super(Loader, self).__init__(workflow, **kwargs)
        self.verify_interface(ILoader)

        self.prng = kwargs.get("prng", random_generator.get())

        if not self.testing:
            self.shuffle_limit = kwargs.get(
                "shuffle_limit", numpy.iinfo(numpy.uint32).max)
        else:
            self.shuffle_limit = 0
        self._max_minibatch_size = kwargs.get("minibatch_size", 100)
        if self._max_minibatch_size < 1:
            raise ValueError("minibatch_size must be greater than zero")

        self._class_lengths = [0] * len(CLASS_NAME)
        self._class_end_offsets = [0] * len(CLASS_NAME)
        self._has_labels = False

        self.epoch_ended = Bool()
        self.epoch_number = 0
        self.train_ended = Bool()
        self.test_ended = Bool()

        self.samples_served = 0
        self._global_offset = 0

        self.minibatch_class = 0
        self.minibatch_data = memory.Array(shallow_pickle=True)
        self.minibatch_indices = memory.Array(shallow_pickle=True)
        self.minibatch_labels = memory.Array(shallow_pickle=True)
        self._raw_minibatch_labels = []
        self._labels_mapping = {}
        self._reversed_labels_mapping = []
        self._samples_mapping = defaultdict(set)

        self.failed_minibatches = []
        self._total_failed = 0
        self._on_initialized = nothing
        self._unique_labels_count = 1  # "None" label

        self.shuffled_indices = memory.Array()
        self.normalization_type = kwargs.get("normalization_type", "none")
        self.normalization_parameters = kwargs.get(
            "normalization_parameters", {})
        self.train_ratio = kwargs.get("train_ratio", self.train_ratio)

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self._minibatch_offset_ = 0
        self._minibatch_size_ = 0
        self.pending_minibatches_ = defaultdict(list)
        self._minibatch_serve_timestamp_ = time.time()
        self.initialize = self._with_initialized_callback(self.initialize)
        parser = Loader.init_parser()
        args, _ = parser.parse_known_args(self.argv)
        self.train_ratio = args.train_ratio

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        # Move all pending minibatches to failed set
        if not self.epoch_ended:
            state["failed_minibatches"] = copy(state["failed_minibatches"])
            for pmb in self.pending_minibatches_.values():
                state["failed_minibatches"].extend(pmb)
        else:
            state["failed_minibatches"] = []
        oni = self._on_initialized
        if oni == nothing:
            state["_on_initialized"] = None
        else:
            state["_on_initialized"] = (
                oni.__name__, marshal.dumps(oni.__code__),
                tuple(c.cell_contents for c in oni.__closure__))
        return state

    def __setstate__(self, state):
        oni_tuple = state.pop("_on_initialized")
        super(Loader, self).__setstate__(state)
        if oni_tuple is not None:
            def cell(obj):
                return (lambda: obj).__closure__[0]

            name, code, closure = oni_tuple
            closure = tuple(cell(c) for c in closure)
            self._on_initialized = \
                types.FunctionType(
                    marshal.loads(code), globals(), name, closure=closure)
        else:
            self._on_initialized = nothing

    def derive_from(self, loader):
        self.normalization_type = loader.normalization_type
        self.normalization_parameters = loader.normalization_parameters
        self._normalizer = loader.normalizer
        self._minibatch_data_shape = (1,) + loader.minibatch_data.shape[1:]
        self._unique_labels_count = loader.unique_labels_count
        self._labels_mapping = loader.labels_mapping

    @property
    def has_labels(self):
        """
        True if the loaded dataset has labels; otherwise, False.
        This is set after initialize() (particularly, after load_data()).
        """
        return self._has_labels

    @property
    def labels_mapping(self):
        """
        :return: dictionary object label -> integer label (internal)
        """
        return self._labels_mapping

    @property
    def reversed_labels_mapping(self):
        """
        :return: dictionary integer label (internal) -> object label
        """
        return self._reversed_labels_mapping

    @property
    def unique_labels_count(self):
        if self._unique_labels_count <= 1 and self.class_lengths[TRAIN] > 0 \
                and self.has_labels:
            different_labels = set()
            self.info("Counting unique labels...")
            self._iterate_class(TRAIN, lambda: different_labels.update(
                self.minibatch_labels))
            self._unique_labels_count = len(different_labels)
        return self._unique_labels_count

    @property
    def _unique_labels_count(self):
        return self.__unique_labels_count

    @_unique_labels_count.setter
    def _unique_labels_count(self, value):
        if self.has_labels:
            self.info("There are %d unique labels", value)
        self.__unique_labels_count = value

    @property
    def samples_mapping(self):
        return self._samples_mapping

    @property
    def on_initialized(self):
        return self._on_initialized

    @on_initialized.setter
    def on_initialized(self, value):
        if not callable(value):
            raise TypeError("on_initialized must be callable")
        self._on_initialized = value

    @property
    def dtype(self):
        return dtypes[config.root.common.engine.precision_type]

    @property
    def normalization_type(self):
        return self._normalization_type

    @normalization_type.setter
    def normalization_type(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Normalization type must be a string (got %s)", type(value))
        if value not in normalization.NormalizerRegistry.normalizers:
            raise ValueError("Unknown normalization type \"%s\"" % value)
        assert not self.is_initialized
        self._normalization_type = value
        self.normalization_parameters = {}

    @property
    def normalization_parameters(self):
        return self._normalization_parameters

    @normalization_parameters.setter
    def normalization_parameters(self, value):
        if not isinstance(value, dict):
            raise TypeError("Normalization parameters must be a dictionary")
        self._normalization_parameters = value
        self._normalizer = None

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = normalization.NormalizerRegistry.normalizers[
                self.normalization_type](**self.normalization_parameters)
        return self._normalizer

    @property
    def class_lengths(self):
        return self._class_lengths

    @property
    def class_end_offsets(self):
        return self._class_end_offsets

    @property
    def effective_class_end_offsets(self):
        return self._effective_class_end_offsets

    @property
    def global_offset(self):
        return self._global_offset

    @global_offset.setter
    def global_offset(self, value):
        if not isinstance(value, int):
            raise TypeError(
                "global_offset must be an integer (got %s)" % type(value))
        if value < 0 or value > self.total_samples:
            raise ValueError(
                "global_offset must be in [0, %d] (got %d)" % (
                    self.total_samples, value))
        self._global_offset = value

    @property
    def shuffled_indices(self):
        return self._shuffled_indices

    @shuffled_indices.setter
    def shuffled_indices(self, value):
        self._shuffled_indices = value

    @property
    def total_samples(self):
        return sum(self.class_lengths)

    @property
    def effective_total_samples(self):
        return self.total_samples - \
            int((1.0 - self.train_ratio) * self.class_lengths[TRAIN])

    @property
    def samples_served(self):
        return self._samples_served

    @samples_served.setter
    def samples_served(self, value):
        self._samples_served = value
        if not self.is_slave and value > 0:
            num, den = divmod(self.samples_served,
                              self.effective_total_samples)
            self.epoch_number = num
            now = time.time()
            if now - self._minibatch_serve_timestamp_ >= 10:
                self._minibatch_serve_timestamp_ = now
                self.info("Served %d samples (%d epochs, %.1f%% current); "
                          "jobs failed: %d; pending: %d",
                          self.samples_served, num,
                          100. * den / self.effective_total_samples,
                          len(self.failed_minibatches),
                          self.pending_minibatches_count)

    @property
    def pending_minibatches_count(self):
        return sum(len(v) for v in self.pending_minibatches_.values())

    @property
    def minibatch_class(self):
        return self._minibatch_class

    @minibatch_class.setter
    def minibatch_class(self, value):
        if not 0 <= value < len(CLASS_NAME):
            raise ValueError("Invalid minibatch_class value %s" % str(value))
        self._minibatch_class = value

    @property
    def minibatch_offset(self):
        return self._minibatch_offset_

    @minibatch_offset.setter
    def minibatch_offset(self, value):
        if not 0 <= value <= self.total_samples:
            raise ValueError("Invalid minibatch_offset value %s" % str(value))
        self._minibatch_offset_ = value
        self._update_flags()

    @property
    def minibatch_size(self):
        return self._minibatch_size_

    @minibatch_size.setter
    def minibatch_size(self, value):
        if not 0 < value <= self.max_minibatch_size:
            raise ValueError("Invalid minibatch_size value %s" % str(value))
        self._minibatch_size_ = value

    @property
    def max_minibatch_size(self):
        return self._max_minibatch_size

    @max_minibatch_size.setter
    def max_minibatch_size(self, value):
        if value < 1:
            raise ValueError("Invalid max_minibatch_size value %s" %
                             str(value))
        self._max_minibatch_size = min(value, max(self.class_lengths))
        if self._max_minibatch_size < 1:
            raise ValueError("max(self.class_lengths) is %d" %
                             max(self.class_lengths))
        self.info("Minibatch size is set to %d", self.max_minibatch_size)

    @property
    def minibatch_data(self):
        return self._minibatch_data

    @minibatch_data.setter
    def minibatch_data(self, value):
        self._minibatch_data = value

    @property
    def minibatch_indices(self):
        return self._minibatch_indices

    @minibatch_indices.setter
    def minibatch_indices(self, value):
        self._minibatch_indices = value

    @property
    def minibatch_labels(self):
        return self._minibatch_labels

    @minibatch_labels.setter
    def minibatch_labels(self, value):
        self._minibatch_labels = value

    @property
    def raw_minibatch_labels(self):
        return self._raw_minibatch_labels

    @property
    def epoch_number(self):
        return self._epoch_number

    @epoch_number.setter
    def epoch_number(self, value):
        if value < 0:
            raise ValueError("epoch_number must be greater than or equal to 0")
        if self.is_master:
            if hasattr(self, "_epoch_number"):
                self.event("epoch", "end",
                           number=self._epoch_number,  # pylint: disable=E0203
                           height=0.25)
            self.event("epoch", "begin", number=value, height=0.25)
        self._epoch_number = value

    @property
    def prng(self):
        """
        Returns the Pseudo Random Number Generator belonging to this instance.
        """
        return self._prng

    @prng.setter
    def prng(self, value):
        if not isinstance(value, random_generator.RandomGenerator):
            raise TypeError("prng must be an instance of RandomGenerator")
        self._prng = value

    @property
    def train_ratio(self):
        return self._train_ratio

    @train_ratio.setter
    def train_ratio(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "train_ratio must be a number (got %s)" % type(value))
        if value <= 0 or value > 1:
            raise ValueError("train_ratio must be in (0, 1] (got %f)" % value)
        self._train_ratio = value

    @property
    def class_ended(self):
        for offset in self.effective_class_end_offsets:
            if self.global_offset == offset:
                return True
            if self.global_offset < offset:
                return False
        raise error.Bug("global_offset %d is out of bounds %s" %
                        (self.global_offset, self.effective_class_end_offsets))

    @property
    def total_failed(self):
        return self._total_failed

    @property
    def shape(self):
        """
        Takes the shape from minibatch_data.
        :return: Sample's shape.
        """
        assert bool(self.minibatch_data), \
            "May be called after create_minibatch_data()"
        return self.minibatch_data[0].shape

    @staticmethod
    def init_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--train-ratio", default=1.0, type=float,
                            help="Use the given fraction of the whole train "
                                 "dataset.")
        return parser

    def initialize(self, **kwargs):
        """Loads the data, initializes indices, shuffles the training set.
        """
        if self.testing:
            self.shuffle_limit = 0
            self.global_offset = 0
            del self.failed_minibatches[:]
        try:
            super(Loader, self).initialize(**kwargs)
        except AttributeError:
            pass
        try:
            self.load_data()
        except AttributeError as e:
            self.exception("Failed to load the data")
            raise from_none(e)
        if self.class_lengths[TRAIN] > 0:
            self.reset_normalization()
        self.max_minibatch_size = kwargs.get("minibatch_size",
                                             self.max_minibatch_size)
        self.on_before_create_minibatch_data()
        self._calc_class_end_offsets()
        sn_log_str = "Samples number: test: %d, validation: %d, train: %d"
        if self.train_ratio == 1.0:
            self.info(sn_log_str, *self.class_lengths)
        else:
            self.info(sn_log_str + " (used: %d)", *(self.class_lengths + [
                self.effective_class_end_offsets[TRAIN] -
                self.effective_class_end_offsets[VALID]]))

        self.minibatch_labels.reset(numpy.zeros(
            self.max_minibatch_size, dtype=Loader.LABEL_DTYPE)
            if self.has_labels else None)
        self.raw_minibatch_labels[:] = (None,) * self.max_minibatch_size
        self.minibatch_indices.reset(numpy.zeros(
            self.max_minibatch_size, dtype=Loader.INDEX_DTYPE))

        try:
            self.create_minibatch_data()
        except Exception as e:
            self.error("Failed to create minibatch data")
            raise from_none(e)

        if not self.minibatch_data:
            raise error.BadFormatError("minibatch_data MUST be initialized in "
                                       "create_minibatch_data()")
        self.analyze_dataset()
        if self.testing:
            self.shuffled_indices.mem = None
        if not self.restored_from_snapshot or self.testing:
            self.shuffle()

    def run(self):
        """Prepares the minibatch.
        """
        if None in self.pending_minibatches_:
            del self.pending_minibatches_[None]
        self.serve_next_minibatch(None)
        self._on_successful_serve()

    def generate_data_for_master(self):
        return True

    def generate_data_for_slave(self, slave):
        self.serve_next_minibatch(slave.id)
        data = {'indices': self.minibatch_indices.mem[:self.minibatch_size]}
        for attr in ("minibatch_class", "minibatch_size", "minibatch_offset",
                     "epoch_number"):
            data[attr] = getattr(self, attr)
        self.has_data_for_slave = ((not self.class_ended) or
                                   len(self.failed_minibatches) > 0)
        return data

    def apply_data_from_master(self, data):
        # Just feed single minibatch
        for attr in ("minibatch_class", "minibatch_size", "minibatch_offset",
                     "epoch_number"):
            setattr(self, attr, data[attr])
        self.last_minibatch <<= False
        self.epoch_ended <<= False
        self.train_ended <<= False
        indices = data['indices']
        if indices.size != self.minibatch_size:
            raise error.MasterSlaveCommunicationError(
                "minibatch size mismatch")
        if self.minibatch_offset > len(self.shuffled_indices):
            raise error.MasterSlaveCommunicationError(
                "minibatch offset overflow")
        if self.minibatch_offset - self.minibatch_size < 0:
            raise error.MasterSlaveCommunicationError(
                "minibatch offset - size < 0")
        # Patch shuffled_indices so that received indices will be picked up
        # during  serve_next_minibatch()
        self.shuffled_indices.map_write()
        self.shuffled_indices.mem[self.minibatch_offset - self.minibatch_size:
                                  self.minibatch_offset] = indices

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            # Partial update
            return
        try:
            self.minibatch_offset, self.minibatch_size = \
                self.pending_minibatches_[slave.id].pop()
        except KeyError:
            raise error.Bug("pending_minibatches_ does not contain %s" %
                            slave.id)
        self._on_successful_serve()
        if not self.has_data_for_slave:
            self.has_data_for_slave = self.last_minibatch

    def drop_slave(self, slave):
        if slave.id in self.pending_minibatches_:
            self._total_failed += 1
            self.failed_minibatches.extend(self.pending_minibatches_[slave.id])
            del self.pending_minibatches_[slave.id]
            self.has_data_for_slave = True
            self.info("Jobs failed: %d/pending: %d",
                      len(self.failed_minibatches),
                      self.pending_minibatches_count)

    def get_metric_names(self):
        if not self.testing:
            return {"Total epochs"}
        if self.has_labels:
            return {"Labels"}
        return set()

    def get_metric_values(self):
        if not self.testing:
            return {"Total epochs": self.epoch_number}
        if self.has_labels:
            return {"Labels": self.reversed_labels_mapping}
        return {}

    def reset_normalization(self):
        self.normalizer.reset()

    def on_before_create_minibatch_data(self):
        self.minibatch_data.reset()
        self.minibatch_labels.reset()
        self.minibatch_indices.reset()

    def shuffle(self):
        """Randomly shuffles the TRAIN dataset.
        """
        if not self.shuffled_indices:
            self.shuffled_indices.mem = numpy.arange(
                self.total_samples, dtype=Loader.INDEX_DTYPE)
        if self.shuffle_limit <= 0 or self.class_lengths[TRAIN] == 0:
            return
        self.shuffle_limit -= 1
        self.debug("Shuffling, remaining limit is %d", self.shuffle_limit)
        self.shuffled_indices.map_write()
        self.prng.shuffle(self.shuffled_indices.mem[
            self.class_end_offsets[VALID]:])
        self.debug("Shuffled %s set", CLASS_NAME[TRAIN])

    def serve_next_minibatch(self, slave_id):
        try:
            minibatch_def = self.failed_minibatches.pop()
        except IndexError:
            minibatch_def = self._advance_global_offset()
        minibatch_offset, minibatch_size = minibatch_def
        self.pending_minibatches_[slave_id].append(minibatch_def)
        self.minibatch_offset, self.minibatch_size = minibatch_def

        if self.fill_indices(minibatch_offset - minibatch_size,
                             minibatch_size):
            # If this method returned True, it means that some acceleration
            # is used and numpy/CPU is not directly used; effectively,
            # fill_minibatch() becomes redundant.
            return

        if self.is_master:
            return

        self.fill_minibatch()
        self.normalize_minibatch()
        self.map_minibatch_labels()

        if minibatch_size < self.max_minibatch_size:
            self.minibatch_data[minibatch_size:] = 0.0
            if self.has_labels:
                self.minibatch_labels[minibatch_size:] = -1
            self.minibatch_indices[minibatch_size:] = -1

    def analyze_dataset(self):
        if self.class_lengths[TRAIN] == 0:
            assert self.normalizer.is_initialized, \
                "There are no train samples and the normalizer has not been " \
                "initialized. Either derive this loader from an existing one" \
                " (use derive_from()) or provide the normalizer's state " \
                "manually (use normalizer.state property)."
            return
        if isinstance(self.normalizer, normalization.StatelessNormalizer):
            self.info('Skipped normalization analysis (type was set to "%s")',
                      type(self.normalizer).MAPPING)
            # Call to analyze() is still needed
            self.normalizer.analyze(self.minibatch_data.mem)
            if self.has_labels and len(self.labels_mapping) == 0:
                raise ValueError("Normalization analysis was skipped but you "
                                 "did not setup labels_mapping in load_data()")
            self._unique_labels_count = len(self.labels_mapping)
            return
        self.info("Performing \"%s\" normalization analysis...",
                  type(self.normalizer).MAPPING)
        train_different_labels = defaultdict(int)

        def callback():
            if self.has_labels and len(self.labels_mapping) == 0:
                for lbl in self.raw_minibatch_labels:
                    train_different_labels[lbl] += 1
            self.normalizer.analyze(self.minibatch_data[:self.minibatch_size])

        self._iterate_class(TRAIN, callback)

        if not self.has_labels or (len(self.labels_mapping) > 0 and
                                   len(self._samples_mapping) > 0):
            return

        other_different_labels = defaultdict(int), defaultdict(int)

        for index, diff_labels in other_different_labels:
            def other_callback():
                for sind, lbl in zip(self.minibatch_indices,
                                     self.raw_minibatch_labels):
                    other_different_labels[index][lbl] += 1  # nopep8 pylint: disable=W0640
                    self._samples_mapping[lbl].add(sind)

            self._iterate_class(index, other_callback)

        if len(self.labels_mapping) == 0:
            self._setup_labels_mapping(
                other_different_labels + (train_different_labels,))

    def normalize_minibatch(self):
        self.normalizer.normalize(self.minibatch_data[:self.minibatch_size])

    def map_minibatch_labels(self):
        if not self.has_labels:
            return
        self.minibatch_labels.map_write()
        for i, l in enumerate(self.raw_minibatch_labels[:self.minibatch_size]):
            try:
                self.minibatch_labels[i] = self.labels_mapping[l]
            except KeyError as e:
                if i == 0 and l is None:
                    self.error(
                        "Looks like you forgot to fill raw_minibatch_labels "
                        "inside fill_minibatch()")
                raise from_none(e)

    def fill_indices(self, start_offset, count):
        """Fills minibatch_indices.

        May fill minibatch data, labels, etc.,
        should return True in such case.

        Returns:
            True: no further processing needed.
            False: only indices were filled, further processing required.
        """
        for v in (self.minibatch_data, self.minibatch_labels,
                  self.minibatch_indices):
            v.map_invalidate()
        self.shuffled_indices.map_read()
        self.minibatch_indices.mem[:count] = self.shuffled_indices[
            start_offset:start_offset + count]
        return False

    def class_index_by_sample_index(self, index):
        for class_index, class_offset in enumerate(
                self.effective_class_end_offsets):
            if index < class_offset:
                return class_index, class_offset - index
        raise error.Bug("Could not convert sample index to class index, "
                        "probably due to incorrect class_end_offsets.")

    def _calc_class_end_offsets(self):
        """Fills self.class_end_offsets from self.class_lengths.
        """
        total_samples = 0
        for i, n in enumerate(self.class_lengths):
            assert isinstance(n, int), \
                "class_length must contain integers only"
            total_samples += n
            self.class_end_offsets[i] = total_samples
        if total_samples == 0:
            raise ValueError("There is no data to serve")
        self._effective_class_end_offsets = list(self.class_end_offsets)
        self._effective_class_end_offsets[TRAIN] -= \
            int((1.0 - self.train_ratio) * self.class_lengths[TRAIN])

    def _update_flags(self):
        """Resets epoch_ended and last_minibatch.
        """
        if self.is_slave:
            # The flags will be explicitly set in apply_data_from_master()
            return
        last_mb = (
            self.class_ended and
            (not self.pending_minibatches_count or not self.is_master) and
            not len(self.failed_minibatches))
        self.last_minibatch <<= last_mb
        self.epoch_ended <<= last_mb and (
            self.minibatch_class == VALID or
            (self.minibatch_class == TEST and self.class_lengths[TRAIN] ==
                self.class_lengths[VALID] == 0) or
            (self.minibatch_class == TEST and self.testing) or
            (self.minibatch_class == TRAIN and self.class_lengths[VALID] == 0))

    def _advance_global_offset(self):
        """Increments global_offset by an appropriate minibatch_size.
        """
        # Slave mode is much simpler than others
        if self.is_slave:
            return self.minibatch_offset, self.minibatch_size
        # Shuffle again when the end of data is reached.
        if self.global_offset >= self.effective_total_samples:
            self.global_offset = 0
            self.shuffle()

        # Compute next minibatch class and size
        self.minibatch_class, remainder = self.class_index_by_sample_index(
            self.global_offset)
        minibatch_size = min(remainder, self.max_minibatch_size)
        self.global_offset += minibatch_size
        self.train_ended <<= self.global_offset >= self.effective_total_samples
        self.test_ended <<= self.global_offset >= self.class_end_offsets[TEST]
        return self.global_offset, minibatch_size

    def _on_successful_serve(self):
        self.samples_served += self.minibatch_size
        if self.last_minibatch:
            self.info("Last minibatch (%d total) of class %s served in epoch "
                      "%d", self.class_lengths[self.minibatch_class],
                      CLASS_NAME[self.minibatch_class].upper(),
                      self.epoch_number)
            # The following line will reduce info message count
            # for small datasets
            self._minibatch_serve_timestamp_ = time.time()

    def _iterate_class(self, class_index, fn):
        size = int(numpy.ceil(
            self.class_lengths[class_index] / self.max_minibatch_size))
        for i in ProgressBar(term_width=40)(range(size)):
            start_index = i * self.max_minibatch_size
            self.minibatch_size = min(
                self.max_minibatch_size,
                self.class_lengths[class_index] - start_index)
            offset = self.class_end_offsets[class_index - 1] + start_index
            self.minibatch_indices[:self.minibatch_size] = \
                self.shuffled_indices[offset:offset + self.minibatch_size]
            self.fill_minibatch()
            fn()

    def _setup_labels_mapping(self, diff_labels):
        if not self.has_labels:
            return
        other_diff_labels, self.train_diff_labels = \
            diff_labels[:TRAIN], diff_labels[TRAIN]
        self.test_diff_labels, self.valid_diff_labels = other_diff_labels
        self._unique_labels_count = len(self.train_diff_labels)
        if len(self.labels_mapping) == 0:
            self.labels_mapping.update(
                {k: i for i, k in enumerate(sorted(self.train_diff_labels))})
            self._reversed_labels_mapping[:] = sorted(self.labels_mapping)
        self._print_label_stats(self.train_diff_labels, CLASS_NAME[TRAIN])
        for i, diff_labels in enumerate(other_diff_labels):
            if self.class_lengths[i] > 0:
                self._validate_and_fix_other_labels(diff_labels)
                self._print_label_stats(diff_labels, CLASS_NAME[i])
        train_dist, test_dist, valid_dist = map(
            self._calc_labels_normalized_distribution,
            (self.train_diff_labels,) + other_diff_labels)
        for i, dist in enumerate((test_dist, valid_dist)):
            self._compare_label_distributions(train_dist, dist, CLASS_NAME[i])

    def _print_label_stats(self, stats, set_name):
        values = list(stats.values())
        if sum(values) == 0:
            self.info("No %s labels specified", set_name)
            return
        mean = int(numpy.mean(values))
        stddev = int(numpy.std(values))
        lmin = numpy.min(values)
        amin = list(stats.keys())[numpy.argmin(values)]
        lmax = numpy.max(values)
        amax = list(stats.keys())[numpy.argmax(values)]
        if has_colors() and stddev > mean / 10:
            endstr = "\033[0m"
            if stddev > mean / 2:
                openstr = "\033[1;31m"  # red
            else:
                openstr = "\033[1;33m"  # yellow
        else:
            openstr = endstr = ""
        self.info(
            u"%s label cardinalities: min: %d (\"%s\"), max: %d (\"%s\"), avg:"
            u" %d, %sσ: %d (%d%%)%s", set_name, lmin, amin, lmax, amax, mean,
            openstr, stddev, stddev * 100 // mean, endstr)
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        total = sum(values)
        table = PrettyTable("Label", "Cardinality", "%", "Histogram")
        table.align["Cardinality"] = "r"
        table.align["%"] = "r"
        table.align["Histogram"] = "l"
        for k, v in stats.items():
            table.add_row(k, v, "%.1f" % (v * 100 / total),
                          "*" * (v * 25 // lmax))
        self.debug("Detailed %s label stats:\n%s", set_name, table)

    @staticmethod
    def _calc_labels_normalized_distribution(different_labels):
        distribution = numpy.array(
            [v for k, v in sorted(different_labels.items())], numpy.float64)
        dist_sum = numpy.sum(distribution)
        if dist_sum > 0:
            distribution /= dist_sum
        return distribution

    def _validate_and_fix_other_labels(self, other_labels):
        train_set = set(self.labels_mapping)
        other_set = set(other_labels)
        diff = other_set - train_set
        if len(diff) > 0:
            self.debug("Other labels: %s", other_labels)
            raise LoaderError(
                "There are no such labels in the training set: %s" % diff)
        diff = train_set - other_set
        if len(diff) > 0:
            for lbl in diff:
                other_labels[lbl] = 0
            self.warning(
                "There are no such labels in the test/validation set: %s",
                diff)

    def _compare_label_distributions(self, train_dist, other_dist, other_name):
        if sum(other_dist) == 0:
            return
        if chisquare is not None:
            _, p = chisquare(other_dist, train_dist)
            is_the_same = p > 0.95
            msg = (CLASS_NAME[TRAIN] + u" and %s labels have %s "
                   u"distributions (Χ-square test's p-value is %.3f)")
            if is_the_same:
                self.info(u"OK: " + msg, other_name, u"the same", p)
            else:
                self.warning(msg, other_name, u"different", p)

    def _with_initialized_callback(self, fn):
        def wrapped_on_initialized(*args, **kwargs):
            retry = fn(*args, **kwargs)
            assert retry is None or isinstance(retry, bool)
            if not retry:
                self.on_initialized()  # pylint: disable=E1102
            return retry

        fnname = getattr(fn, '__name__',
                         getattr(fn, 'func', wrapped_on_initialized).__name__)
        wrapped_on_initialized.__name__ = fnname + '_on_initialized'
        return wrapped_on_initialized


class LoaderMSEMixin(Unit):
    hide_from_registry = True
    """
    Loader MSE implementation for parallel inheritance.

    Attributes:
        class_targets: target for each class.
        minibatch_targets: target data.
    """

    def __init__(self, workflow, **kwargs):
        super(LoaderMSEMixin, self).__init__(workflow, **kwargs)
        self.class_targets = memory.Array()
        self._minibatch_targets = memory.Array(shallow_pickle=True)
        self._targets_shape = kwargs.get("targets_shape", tuple())
        self.target_normalization_type = kwargs.get(
            "target_normalization_type",
            kwargs.get("normalization_type", "none"))
        if "target_normalization_type" in kwargs and \
                self.target_normalization_type != self.normalization_type and \
                "target_normalization_parameters" not in kwargs:
            raise ValueError("You set target_normalization_type in %s which "
                             "is different from normalization_type but did not"
                             " set target_normalization_parameters." %
                             self.target_normalization_type)
        self.target_normalization_parameters = kwargs.get(
            "target_normalization_parameters",
            kwargs.get("normalization_parameters", {}))

    @property
    def targets_shape(self):
        return self._targets_shape

    @property
    def target_normalization_type(self):
        return self._target_normalization_type

    @target_normalization_type.setter
    def target_normalization_type(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Normalization type must be a string (got %s)", type(value))
        if value not in normalization.NormalizerRegistry.normalizers:
            raise ValueError("Unknown normalization type \"%s\"" % value)
        assert not self.is_initialized
        self._target_normalization_type = value
        self.target_normalization_parameters = {}

    @property
    def target_normalization_parameters(self):
        return self._target_normalization_parameters

    @target_normalization_parameters.setter
    def target_normalization_parameters(self, value):
        if not isinstance(value, dict):
            raise TypeError("Normalization parameters must be a dictionary")
        self._target_normalization_parameters = value
        self._target_normalizer = None

    @property
    def target_normalizer(self):
        nr = normalization.NormalizerRegistry
        if self._target_normalizer is None:
            self._target_normalizer = nr.normalizers[
                self.target_normalization_type](
                **self.target_normalization_parameters)
            if isinstance(self._target_normalizer,
                          normalization.StatelessNormalizer) and \
                    not isinstance(self._target_normalizer,
                                   normalization.NoneNormalizer):
                raise AttributeError(
                    "The specified normalization type \"%s\" is stateless, "
                    "that is, there is no way do denormalize the output "
                    "without knowing the original data traits. Stateless "
                    "normalizers are restricted in MSE mode since the forward "
                    "propagation on test data becomes impossible." %
                    self.target_normalization_type)
        return self._target_normalizer

    @property
    def minibatch_targets(self):
        return self._minibatch_targets

    def initialize(self, **kwargs):
        super(LoaderMSEMixin, self).initialize(**kwargs)
        if self.class_lengths[TRAIN] > 0:
            self.target_normalizer.reset()

    def analyze_dataset(self):
        super(LoaderMSEMixin, self).analyze_dataset()
        if self.targets_shape == tuple():
            self._targets_shape = self.minibatch_targets.shape[1:]
        elif numpy.prod(self.targets_shape) != numpy.prod(
                self.minibatch_targets.shape[1:]):
            raise ValueError(
                "targets_shape is %s but minibatch_targets has shape %s: "
                "products do not match" % (self.targets_shape,
                                           self.minibatch_targets.shape[1:]))
        self.info("Target shape is set to %s", self.targets_shape)

    def on_before_create_minibatches(self):
        super(LoaderMSEMixin, self).on_before_create_minibatches()
        self.minibatch_targets.reset()

    def serve_next_minibatch(self, slave):
        super(LoaderMSEMixin, self).serve_next_minibatch(slave)

        if self.minibatch_size < self.max_minibatch_size:
            self.minibatch_targets[self.minibatch_size:] = 0.0

    def fill_indices(self, start_offset, count):
        self.minibatch_targets.map_invalidate()
        return super(LoaderMSEMixin, self).fill_indices(start_offset, count)


class LoaderMSE(Loader, LoaderMSEMixin):
    """Loader with MSE target data.
    Attributes:
        class_targets: target for each class.
        minibatch_targets: target data.
    """
    pass


class LoaderWithValidationRatio(Loader):
    def __init__(self, workflow, **kwargs):
        super(LoaderWithValidationRatio, self).__init__(workflow, **kwargs)
        self.validation_ratio = kwargs.get("validation_ratio", None)

    @property
    def validation_ratio(self):
        return getattr(self, "_validation_ratio", None)

    @validation_ratio.setter
    def validation_ratio(self, value):
        if value is None:
            self._validation_ratio = None
            return
        if isinstance(value, int):
            if value != 0:
                raise ValueError("validation_ratio must be in [0, 1).")
            self._validation_ratio = 0.0
            return
        if not isinstance(value, float):
            raise TypeError("validation_ratio must be a float")
        if value < 0 or value >= 1:
            raise ValueError("validation_ratio must be in [0, 1).")
        self._validation_ratio = value
