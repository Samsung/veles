"""
Created on Aug 14, 2013

FullBatchLoader class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division

import numpy
import six
from zope.interface import implementer, Interface

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit
import veles.error as error
import veles.memory as memory
from veles.opencl_types import numpy_dtype_to_opencl
from veles.units import UnitCommandLineArgumentsRegistry
from veles.loader.base import ILoader, Loader, LoaderMSEMixin, \
    UserLoaderRegistry


TRAIN = 2
VALID = 1
TEST = 0


class FullBatchUserLevelLoaderRegistry(UnitCommandLineArgumentsRegistry,
                                       UserLoaderRegistry):
    pass


@six.add_metaclass(FullBatchUserLevelLoaderRegistry)
class FullBatchLoaderBase(Loader):
    pass


class IFullBatchLoader(Interface):
    def load_data():
        """Load the data here.
        """


@implementer(ILoader, IOpenCLUnit)
class FullBatchLoader(AcceleratedUnit, FullBatchLoaderBase):
    """Loads data entire in memory.

    Attributes:
        original_data: original data (Vector).
        original_labels: original labels (Vector, dtype=Loader.LABEL_DTYPE)
            (in case of classification).
        on_device: True to load all data to the device memory.

    Should be overriden in child class:
        load_data()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchLoader, self).__init__(workflow, **kwargs)
        self.verify_interface(IFullBatchLoader)
        self.on_device = kwargs.get("on_device", False)
        self.validation_ratio = kwargs.get("validation_ratio", None)

    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self.original_data = memory.Vector()
        self.original_labels = memory.Vector()
        self.cl_sources_["fullbatch_loader"] = {}
        self._global_size = None
        self._krn_const = numpy.zeros(2, dtype=Loader.LABEL_DTYPE)

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        for attr in "original_data", "original_labels":
            state[attr] = None
        return state

    @Loader.shape.getter
    def shape(self):
        """
        Takes the shape from original_data.
        :return: Sample's shape.
        """
        if not self.original_data:
            raise AttributeError("Must first initialize original_data")
        return self.original_data[0].shape

    @property
    def validation_ratio(self):
        """
        Returns the ratio between new train and new validation set lengths.
        None means no validation set extraction.
        Negative means move validation to train.
        """
        return self._validation_ratio

    @validation_ratio.setter
    def validation_ratio(self, value):
        if value is None:
            self._validation_ratio = None
            return
        if not isinstance(value, float):
            raise TypeError(
                "validation_ratio must be a floating point value (got %s of "
                "type %s)" % (value, value.__class__))
        if value >= 1:
            raise ValueError(
                "validation_ratio = %f is out of the allowed range (0, 1)" %
                value)
        self._validation_ratio = value

    def check_types(self):
        if (not isinstance(self.original_data, memory.Vector) or
                not isinstance(self.original_labels, memory.Vector)):
            raise error.BadFormatError(
                "original_data, original_labels must be of type Vector")
        if (self.original_labels.mem is not None and
                self.original_labels.dtype != Loader.LABEL_DTYPE):
            raise error.BadFormatError(
                "original_labels should have dtype=Loader.LABEL_DTYPE")

    def get_ocl_defines(self):
        """Add definitions before building the kernel during initialize().
        """
        return {}

    def initialize(self, device, **kwargs):
        super(FullBatchLoader, self).initialize(device=device, **kwargs)
        assert self.total_samples > 0
        self.check_types()

        self.info("Normalizing to %s...", self.normalization_type)
        self.analyze_and_normalize_original_data()

        if not self.on_device or self.device is None:
            return

        self.info("Will load the entire dataset on device")
        self.original_data.initialize(self.device)
        self.minibatch_data.initialize(self.device)
        if self.has_labels:
            self.original_labels.initialize(self.device)
            self.minibatch_labels.initialize(self.device)

        if not self.shuffled_indices:
            self.shuffled_indices.mem = numpy.arange(
                self.total_samples, dtype=Loader.LABEL_DTYPE)
        self.init_vectors(self.shuffled_indices, self.minibatch_indices)

    def _gpu_init(self):
        defines = {
            "LABELS": int(self.has_labels),
            "SAMPLE_SIZE": self.original_data.sample_size,
            "MAX_MINIBATCH_SIZE": self.max_minibatch_size,
            "original_data_dtype": numpy_dtype_to_opencl(
                self.original_data.dtype),
            "minibatch_data_dtype": numpy_dtype_to_opencl(
                self.minibatch_data.dtype)
        }
        defines.update(self.get_ocl_defines())

        self.build_program(defines, "fullbatch_loader",
                           dtype=self.minibatch_data.dtype)
        self.assign_kernel("fill_minibatch_data_labels")

        if not self.has_labels:
            self.set_args(self.original_data, self.minibatch_data,
                          self.device.skip(2),
                          self.shuffled_indices, self.minibatch_indices)
        else:
            self.set_args(self.original_data, self.minibatch_data,
                          self.device.skip(2),
                          self.original_labels, self.minibatch_labels,
                          self.shuffled_indices, self.minibatch_indices)

    def cpu_run(self):
        Loader.run(self)

    def ocl_run(self):
        self.cpu_run()

    def cuda_run(self):
        self.cpu_run()

    def ocl_init(self):
        self._gpu_init()
        self._global_size = (self.max_minibatch_size,
                             self.minibatch_data.sample_size)
        self._local_size = None

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (int(numpy.ceil(
            self.minibatch_data.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def on_before_create_minibatch_data(self):
        self.has_labels = bool(self.original_labels)
        try:
            super(FullBatchLoader, self).on_before_create_minibatch_data()
        except AttributeError:
            pass
        if self.validation_ratio is not None:
            self.resize_validation(ratio=self.validation_ratio)

    def create_minibatch_data(self):
        self.check_types()

        self.minibatch_data.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype))

    def create_originals(self, dshape):
        """
        Create original_data.mem and original_labels.mem.
        :param dshape: Future original_data.shape[1:]
        """
        length = sum(self.class_lengths)
        self.original_data.mem = numpy.zeros((length,) + dshape, self.dtype)
        self.original_labels.mem = numpy.zeros(length, Loader.LABEL_DTYPE)

    def fill_indices(self, start_offset, count):
        if not self.on_device or self.device is None:
            return super(FullBatchLoader, self).fill_indices(
                start_offset, count)

        self.original_data.unmap()
        self.minibatch_data.unmap()

        if self.has_labels:
            self.original_labels.unmap()
            self.minibatch_labels.unmap()

        self.shuffled_indices.unmap()
        self.minibatch_indices.unmap()

        self._krn_const[0:2] = start_offset, count
        self._kernel_.set_arg(2, self._krn_const[0:1])
        self._kernel_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(self._global_size, self._local_size)

        # No further processing needed, so return True
        return True

    def fill_minibatch(self):
        for i, sample_index in enumerate(
                self.minibatch_indices.mem[:self.minibatch_size]):
            self.minibatch_data[i] = self.original_data[sample_index]
            if self.has_labels:
                self.minibatch_labels[i] = self.original_labels[sample_index]

    def analyze_train_for_normalization(self):
        """
        Override.
        """
        pass

    def normalize_minibatch(self):
        """
        Override.
        """
        pass

    def analyze_and_normalize_original_data(self):
        self.normalizer.analyze(self.original_data[
                                self.class_end_offsets[VALID]:])
        self.normalizer.normalize(self.original_data.mem)

    def resize_validation(self, rand=None, ratio=None):
        """Extracts validation dataset from joined validation and train
        datasets randomly.

        We will rearrange indexes only.

        Parameters:
            ratio: how many samples move to validation dataset
                   relative to the entire samples count of validation and
                   train classes.
            rand: veles.prng.RandomGenerator, if it is None, will use self.prng
        """
        rand = rand or self.prng
        if ratio <= 0:  # Dispose validation set
            self.class_lengths[TRAIN] += self.class_lengths[VALID]
            self.class_lengths[VALID] = 0
            if self.shuffled_indices.mem is None:
                total_samples = numpy.sum(self.class_lengths)
                self.shuffled_indices.mem = numpy.arange(
                    total_samples, dtype=Loader.LABEL_DTYPE)
            return
        offs_test = self.class_lengths[TEST]
        offs = offs_test
        train_samples = self.class_lengths[VALID] + self.class_lengths[TRAIN]
        total_samples = train_samples + offs
        if isinstance(self.original_labels, memory.Vector):
            self.original_labels.map_read()
            original_labels = self.original_labels.mem
        else:
            original_labels = self.original_labels

        if self.shuffled_indices.mem is None:
            self.shuffled_indices.mem = numpy.arange(
                total_samples, dtype=Loader.LABEL_DTYPE)
        self.shuffled_indices.map_write()
        shuffled_indices = self.shuffled_indices.mem

        # If there are no labels
        if original_labels is None:
            n = int(numpy.round(ratio * train_samples))
            while n > 0:
                i = rand.randint(offs, offs + train_samples)

                # Swap indexes
                shuffled_indices[offs], shuffled_indices[i] = \
                    shuffled_indices[i], shuffled_indices[offs]

                offs += 1
                n -= 1
            self.class_lengths[VALID] = offs - offs_test
            self.class_lengths[TRAIN] = \
                total_samples - self.class_lengths[VALID] - offs_test
            return

        # If there are labels
        nn = {}
        for i in shuffled_indices[offs:]:
            l = original_labels[i]
            nn[l] = nn.get(l, 0) + 1
        n = 0
        for l in nn.keys():
            n_train = nn[l]
            nn[l] = max(int(numpy.round(ratio * nn[l])), 1)
            if nn[l] >= n_train:
                raise error.NotExistsError("There are too few labels "
                                           "for class %d" % (l))
            n += nn[l]
        while n > 0:
            i = rand.randint(offs, offs_test + train_samples)
            l = original_labels[shuffled_indices[i]]
            if nn[l] <= 0:
                # Move unused label to the end

                # Swap indexes
                ii = shuffled_indices[offs_test + train_samples - 1]
                shuffled_indices[
                    offs_test + train_samples - 1] = shuffled_indices[i]
                shuffled_indices[i] = ii

                train_samples -= 1
                continue
            # Swap indexes
            ii = shuffled_indices[offs]
            shuffled_indices[offs] = shuffled_indices[i]
            shuffled_indices[i] = ii

            nn[l] -= 1
            n -= 1
            offs += 1
        self.class_lengths[VALID] = offs - offs_test
        self.class_lengths[TRAIN] = (total_samples - self.class_lengths[VALID]
                                     - offs_test)


class FullBatchLoaderMSEMixin(LoaderMSEMixin):
    """FullBatchLoader for MSE workflows.
    Attributes:
        original_targets: original target (Vector).
    """
    def init_unpickled(self):
        super(FullBatchLoaderMSEMixin, self).init_unpickled()
        self.original_targets = memory.Vector()
        self._kernel_target_ = None
        self._global_size_target = None

    def __getstate__(self):
        state = super(FullBatchLoaderMSEMixin, self).__getstate__()
        state["original_targets"] = None
        return state

    def create_minibatch_data(self):
        super(FullBatchLoaderMSEMixin, self).create_minibatch_data()
        self.minibatch_targets.reset()
        self.minibatch_targets.mem = numpy.zeros(
            (self.max_minibatch_size,) + self.original_targets[0].shape,
            self.dtype)

    def check_types(self):
        super(FullBatchLoaderMSEMixin, self).check_types()
        if not isinstance(self.original_targets, memory.Vector):
            raise error.BadFormatError(
                "original_targets must be of type Vector")

    def get_ocl_defines(self):
        return {
            "TARGET": 1,
            "TARGET_SIZE": self.original_targets.sample_size,
            "original_target_dtype": numpy_dtype_to_opencl(
                self.original_targets.dtype),
            "minibatch_target_dtype": numpy_dtype_to_opencl(
                self.minibatch_targets.dtype)
        }

    def _gpu_init(self):
        super(FullBatchLoaderMSEMixin, self)._gpu_init()

        self.original_targets.initialize(self.device)
        self.minibatch_targets.initialize(self.device)

        self._kernel_target_ = self.get_kernel("fill_minibatch_target")
        self._kernel_target_.set_args(
            self.original_targets.devmem, self.minibatch_targets.devmem,
            self.device.skip(2), self.shuffled_indices.devmem)

    def ocl_init(self):
        super(FullBatchLoaderMSEMixin, self).ocl_init()
        self._global_size_target = [self.max_minibatch_size,
                                    self.minibatch_targets.sample_size]
        self._local_size_target = None

    def cuda_init(self):
        super(FullBatchLoaderMSEMixin, self).cuda_init()
        block_size = self.device.suggest_block_size(self._kernel_target_)
        self._global_size_target = (int(numpy.ceil(
            self.minibatch_targets.size / block_size)), 1, 1)
        self._local_size_target = (block_size, 1, 1)

    def fill_indices(self, start_offset, count):
        if not super(FullBatchLoaderMSEMixin, self).fill_indices(
                start_offset, count):
            return False
        self.original_targets.unmap()
        self.minibatch_targets.unmap()
        self._kernel_target_.set_arg(2, self._krn_const[0:1])
        self._kernel_target_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(
            self._global_size_target, self._local_size_target,
            self._kernel_target_)
        return True

    def fill_minibatch(self):
        super(FullBatchLoaderMSEMixin, self).fill_minibatch()
        for i, v in enumerate(self.minibatch_indices[:self.minibatch_size]):
            self.minibatch_targets[i] = self.original_targets[v]


class FullBatchLoaderMSE(FullBatchLoaderMSEMixin, FullBatchLoader):
    """FullBatchLoader for MSE workflows.
    """
    pass
