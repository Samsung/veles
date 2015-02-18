"""
Created on Aug 14, 2013

Loader base class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division
from collections import defaultdict
from copy import copy
import numpy
import time
import six
from zope.interface import implementer, Interface

from veles.compat import from_none
import veles.config as config
from veles.distributable import IDistributable
import veles.error as error
from veles.external.progressbar import ProgressBar
import veles.memory as memory
from veles.mutable import Bool
import veles.normalization as normalization
from veles.opencl_types import dtypes
import veles.prng as random_generator
from veles.units import Unit, IUnit
from veles.unit_registry import UnitRegistry

TARGET = 3
TRAIN = 2
VALID = 1
TEST = 0
TRIAGE = {"train": TRAIN,
          "validation": VALID,
          "valid": VALID,
          "test": TEST}
CLASS_NAME = ["test", "validation", "train"]


class UserLoaderRegistry(UnitRegistry):
    loaders = {}

    def __init__(cls, name, bases, clsdict):
        yours = set(cls.mro())
        mine = set(Unit.mro())
        left = yours - mine
        if len(left) > 1 and "MAPPING" in clsdict:
            UserLoaderRegistry.loaders[clsdict["MAPPING"]] = cls
        super(UserLoaderRegistry, cls).__init__(name, bases, clsdict)


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


@implementer(IDistributable, IUnit)
@six.add_metaclass(UserLoaderRegistry)
class Loader(Unit):
    """Loads data and provides minibatch output interface.

    Attributes:
        prng: veles.prng.RandomGenerator instance.
        validation_ratio: used by extract_validation_from_train() as a default
                          ratio.
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

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "LOADER"
        self.last_minibatch = Bool(False)
        super(Loader, self).__init__(workflow, **kwargs)
        self.verify_interface(ILoader)

        self.prng = kwargs.get("prng", random_generator.get())

        self.shuffle_limit = kwargs.get(
            "shuffle_limit", numpy.iinfo(numpy.uint32).max)
        self._max_minibatch_size = kwargs.get("minibatch_size", 100)
        if self._max_minibatch_size < 1:
            raise ValueError("minibatch_size must be greater than zero")

        self._total_samples = 0
        self.class_lengths = [0, 0, 0]
        self.class_end_offsets = [0, 0, 0]
        self._has_labels = False

        self.epoch_ended = Bool(False)
        self.epoch_number = 0
        self.train_ended = Bool(False)

        self.samples_served = 0
        self.global_offset = 0

        self.minibatch_class = 0
        self.minibatch_data = memory.Vector()
        self.minibatch_indices = memory.Vector()
        self.minibatch_labels = memory.Vector()

        self.failed_minibatches = []
        self._total_failed = 0
        self._unpickled = False
        self._on_unique_labels_counted = self.nothing
        self._unique_labels_count = 0

        self.shuffled_indices = memory.Vector()
        self.normalization_type = kwargs.get("normalization_type", "none")
        self.normalization_parameters = kwargs.get(
            "normalization_parameters", {})

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self._minibatch_offset_ = 0
        self._minibatch_size_ = 0
        self.pending_minibatches_ = defaultdict(list)
        self._minibatch_serve_timestamp_ = time.time()
        self._unpickled = True

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        # Move all pending minibatches to failed set
        if not self.epoch_ended:
            state["failed_minibatches"] = copy(state["failed_minibatches"])
            for pmb in self.pending_minibatches_.values():
                state["failed_minibatches"].extend(pmb)
        else:
            state["failed_minibatches"] = []
        return state

    @property
    def has_labels(self):
        """
        True if the loaded dataset has labels; otherwise, False.
        This is set after initialize() (particularly, after load_data()).
        """
        return self._has_labels

    @property
    def unique_labels_count(self):
        if self._unique_labels_count == 0 and self.class_lengths[TRAIN] > 0 \
                and self.has_labels:
            different_labels = set()
            self.info("Counting unique labels...")
            self._iterate_train(lambda: different_labels.update(
                self.minibatch_labels))
            self._unique_labels_count = len(different_labels)
        return self._unique_labels_count

    @property
    def _unique_labels_count(self):
        return self.__unique_labels_count

    @_unique_labels_count.setter
    def _unique_labels_count(self, value):
        self.info("There are %d unique labels", value)
        self.__unique_labels_count = value
        self.on_unique_labels_counted()  # pylint: disable=E1102

    @property
    def on_unique_labels_counted(self):
        return self._on_unique_labels_counted

    @on_unique_labels_counted.setter
    def on_unique_labels_counted(self, value):
        if not hasattr(value, "__call__"):
            raise TypeError("on_unique_labels_counted must be callable")
        self._on_unique_labels_counted = value

    @property
    def dtype(self):
        return dtypes[config.root.common.precision_type]

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
    def shuffled_indices(self):
        return self._shuffled_indices

    @shuffled_indices.setter
    def shuffled_indices(self, value):
        self._shuffled_indices = value

    @property
    def total_samples(self):
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        if value <= 0:
            raise error.BadFormatError("class_lengths should be filled")
        if value > numpy.iinfo(Loader.LABEL_DTYPE).max:
            raise NotImplementedError(
                "total_samples exceeds int32 capacity.")
        self._total_samples = value

    @property
    def samples_served(self):
        return self._samples_served

    @samples_served.setter
    def samples_served(self, value):
        self._samples_served = value
        if not self.is_slave and value > 0:
            num, den = divmod(self.samples_served, self.total_samples)
            self.epoch_number = num
            now = time.time()
            if now - self._minibatch_serve_timestamp_ >= 10:
                self._minibatch_serve_timestamp_ = now
                self.info("Served %d samples (%d epochs, %.1f%% current); "
                          "jobs failed: %d/pending: %d",
                          self.samples_served, num,
                          100. * den / self.total_samples,
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
    def validation_ratio(self):
        return self._validation_ratio

    @validation_ratio.setter
    def validation_ratio(self, value):
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

    @property
    def class_ended(self):
        for offset in self.class_end_offsets:
            if self.global_offset == offset:
                return True
            if self.global_offset < offset:
                return False
        raise error.Bug("global_offset %d is out of bounds %s" %
                        (self.global_offset, self.class_end_offsets))

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

    def initialize(self, **kwargs):
        """Loads the data, initializes indices, shuffles the training set.
        """
        self.normalizer.reset()
        try:
            super(Loader, self).initialize(**kwargs)
        except AttributeError:
            pass
        try:
            self.load_data()
        except AttributeError as e:
            self.exception("Failed to load the data")
            raise from_none(e)
        self.max_minibatch_size = kwargs.get("minibatch_size",
                                             self.max_minibatch_size)
        self.on_before_create_minibatch_data()
        self._update_total_samples()
        self.info("Samples number: test: %d, validation: %d, train: %d",
                  *self.class_lengths)

        self.minibatch_labels.reset(numpy.zeros(
            self.max_minibatch_size, dtype=Loader.LABEL_DTYPE)
            if self.has_labels else None)
        self.minibatch_indices.reset(numpy.zeros(
            self.max_minibatch_size, dtype=Loader.INDEX_DTYPE))

        self.create_minibatch_data()

        if not self.minibatch_data:
            raise error.BadFormatError("minibatch_data MUST be initialized in "
                                       "create_minibatch_data()")
        self.analyze_train_for_normalization()
        if not self._unpickled:
            self.shuffle()
        else:
            self._unpickled = False

    def run(self):
        """Prepares the minibatch.
        """
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

    def on_before_create_minibatch_data(self):
        self.minibatch_data.reset()
        self.minibatch_labels.reset()
        self.minibatch_indices.reset()

    def shuffle(self):
        """Randomly shuffles the TRAIN dataset.
        """
        if self.shuffle_limit <= 0:
            return
        self.shuffle_limit -= 1
        self.debug("Shuffling, remaining limit is %d", self.shuffle_limit)
        if self.shuffled_indices.mem is None:
            self.shuffled_indices.mem = numpy.arange(self.total_samples,
                                                     dtype=Loader.INDEX_DTYPE)
        self.shuffled_indices.map_write()
        self.prng.shuffle(self.shuffled_indices.mem[
            self.class_end_offsets[VALID]:])
        self.debug("Shuffled TRAIN")

    def serve_next_minibatch(self, slave_id):
        try:
            minibatch_offset, minibatch_size = self.failed_minibatches.pop()
        except IndexError:
            minibatch_offset, minibatch_size = self._advance_global_offset()
        if self.is_master:
            self.pending_minibatches_[slave_id].append(
                (minibatch_offset, minibatch_size))
        self.minibatch_offset, self.minibatch_size = \
            minibatch_offset, minibatch_size

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

        if minibatch_size < self.max_minibatch_size:
            self.minibatch_data[minibatch_size:] = 0.0
            self.minibatch_labels[minibatch_size:] = -1
            self.minibatch_indices[minibatch_size:] = -1

    def analyze_train_for_normalization(self):
        if isinstance(self.normalizer, normalization.StatelessNormalizer):
            self.info('Skipped normalization analysis (type was set to "%s")',
                      type(self.normalizer).NAME)
            # Call to analyze() is still needed
            self.normalizer.analyze(self.minibatch_data.mem)
            return
        self.info("Performing \"%s\" normalization analysis...",
                  type(self.normalizer).NAME)
        different_labels = set()

        def callback():
            different_labels.update(self.minibatch_labels)
            self.normalizer.analyze(self.minibatch_data[:self.minibatch_size])

        self._iterate_train(callback)
        self._unique_labels_count = len(different_labels)

    def normalize_minibatch(self):
        self.normalizer.normalize(self.minibatch_data[:self.minibatch_size])

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
        self.minibatch_indices.mem[:count] = self.shuffled_indices.mem[
            start_offset:start_offset + count]
        return False

    def class_index_by_sample_index(self, index):
        for class_index, class_offset in enumerate(self.class_end_offsets):
            if index < class_offset:
                return class_index, class_offset - index
        raise error.Bug("Could not convert sample index to class index, "
                        "probably due to incorrect class_end_offsets.")

    def _update_total_samples(self):
        """Fills self.class_end_offsets from self.class_lengths.
        """
        total_samples = 0
        for i, n in enumerate(self.class_lengths):
            assert isinstance(n, int), \
                "class_length must contain integers only"
            total_samples += n
            self.class_end_offsets[i] = total_samples
        self.total_samples = total_samples
        if self.class_lengths[TRAIN] < 1:
            raise ValueError(
                "class_length for TRAIN dataset is invalid: %d" %
                self.class_lengths[TRAIN])

    def _update_flags(self):
        """Resets epoch_ended and last_minibatch.
        """
        if self.is_slave:
            # The flags will be explicitly set in apply_data_from_master()
            return
        last_mb = (self.class_ended and
                   not self.pending_minibatches_count and
                   not len(self.failed_minibatches))
        self.last_minibatch <<= last_mb
        self.epoch_ended <<= last_mb and (
            self.minibatch_class == VALID or
            (self.minibatch_class == TRAIN and self.class_lengths[VALID] == 0))

    def _advance_global_offset(self):
        """Increments global_offset by an appropriate minibatch_size.
        """
        # Slave mode is much simpler than others
        if self.is_slave:
            return self.minibatch_offset, self.minibatch_size
        # Shuffle again when the end of data is reached.
        if self.global_offset >= self.total_samples:
            self.shuffle()
            self.global_offset = 0

        # Compute next minibatch class and size, updating epoch_ended and
        # last_minibatch
        self.minibatch_class, remainder = self.class_index_by_sample_index(
            self.global_offset)
        minibatch_size = min(remainder, self.max_minibatch_size)
        self.global_offset += minibatch_size
        self.train_ended <<= self.global_offset >= self.total_samples
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

    def _iterate_train(self, fn):
        size = int(numpy.ceil(self.class_lengths[TRAIN] /
                              self.max_minibatch_size))
        for i in ProgressBar(term_width=40)(range(size)):
            start_index = i * self.max_minibatch_size
            self.minibatch_size = min(
                self.max_minibatch_size,
                self.class_lengths[TRAIN] - start_index)
            self.minibatch_indices[:self.minibatch_size] = \
                numpy.arange(start_index, start_index + self.minibatch_size,
                             dtype=numpy.int64) + self.class_end_offsets[VALID]
            self.fill_minibatch()
            fn()


class LoaderMSEMixin(Unit):
    """
    Loader MSE implementation for parallel inheritance.

    Attributes:
        class_targets: target for each class.
        minibatch_targets: target data.
    """

    def __init__(self, workflow, **kwargs):
        super(LoaderMSEMixin, self).__init__(workflow, **kwargs)
        self.class_targets = memory.Vector()
        self.minibatch_targets = memory.Vector()
        self.targets_shape = kwargs.get("targets_shape")
        self.target_normalization_type = kwargs.get(
            "target_normalization_type", "none")
        self.target_normalization_parameters = kwargs.get(
            "target_normalization_parameters", {})

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
        return self._target_normalizer

    @property
    def minibatch_targets(self):
        return self._minibatch_target

    @minibatch_targets.setter
    def minibatch_targets(self, value):
        self._minibatch_target = value

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
