# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 14, 2013

FullBatchLoader class

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
from collections import Counter
import numpy
from cuda4py import CUDARuntimeError, CUDA_ERROR_OUT_OF_MEMORY
from opencl4py import CLRuntimeError, CL_MEM_OBJECT_ALLOCATION_FAILURE
import six
from zope.interface import implementer, Interface

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit, \
    INumpyUnit
from veles.backends import NumpyDevice
from veles.compat import from_none
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


@implementer(ILoader, IOpenCLUnit, ICUDAUnit, INumpyUnit)
class FullBatchLoader(AcceleratedUnit, FullBatchLoaderBase):
    """Loads data entire in memory.

    Attributes:
        original_data: original data (Array).
        original_labels: original labels (Array, dtype=Loader.LABEL_DTYPE)
            (in case of classification).

    Should be overriden in child class:
        load_data()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchLoader, self).__init__(workflow, **kwargs)
        self.verify_interface(IFullBatchLoader)
        self.validation_ratio = kwargs.get("validation_ratio", 0)

    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self._original_data_ = memory.Array()
        self._original_labels_ = []
        self._mapped_original_labels_ = memory.Array()
        self.sources_["fullbatch_loader"] = {}
        self._global_size = None
        self._krn_const = numpy.zeros(2, dtype=Loader.LABEL_DTYPE)

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
    def on_device(self):
        return not self.force_numpy

    @on_device.setter
    def on_device(self, value):
        if not isinstance(value, bool):
            raise TypeError("on_device must be boolean (got %s)" % type(value))
        self.force_numpy = not value

    @property
    def original_data(self):
        return self._original_data_

    @property
    def original_labels(self):
        return self._original_labels_

    def get_ocl_defines(self):
        """Add definitions before building the kernel during initialize().
        """
        return {}

    def initialize(self, device, **kwargs):
        super(FullBatchLoader, self).initialize(device=device, **kwargs)
        assert self.total_samples > 0
        self.analyze_original_dataset()
        self._map_original_labels()

        if isinstance(self.device, NumpyDevice):
            return

        self.info("Will try to store the entire dataset on the device")
        try:
            self.init_vectors(self.original_data, self.minibatch_data)
        except CLRuntimeError as e:
            if e.code == CL_MEM_OBJECT_ALLOCATION_FAILURE:
                self.warning("Failed to store the entire dataset on the "
                             "device")
                self.force_numpy = True
                self.device = NumpyDevice()
                return
            else:
                raise from_none(e)
        except CUDARuntimeError as e:
            if e.code == CUDA_ERROR_OUT_OF_MEMORY:
                self.warning("Failed to store the entire dataset on the "
                             "device")
                self.force_numpy = True
                self.device = NumpyDevice()
                return
            else:
                raise from_none(e)
        if self.has_labels:
            self.init_vectors(self._mapped_original_labels_,
                              self.minibatch_labels)

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
                          self._mapped_original_labels_, self.minibatch_labels,
                          self.shuffled_indices, self.minibatch_indices)

    def _after_backend_init(self):
        try:
            self.fill_indices(0, min(self.max_minibatch_size,
                                     self.total_samples))
        except CLRuntimeError as e:
            if e.code == CL_MEM_OBJECT_ALLOCATION_FAILURE:
                self.warning("Failed to store the entire dataset on the "
                             "device")
                self.force_numpy = True
                self.device = NumpyDevice()
            else:
                raise from_none(e)
        except CUDARuntimeError as e:
            if e.code == CUDA_ERROR_OUT_OF_MEMORY:
                self.warning("Failed to store the entire dataset on the "
                             "device")
                self.force_numpy = True
                self.device = NumpyDevice()
            else:
                raise from_none(e)

    def numpy_run(self):
        Loader.run(self)

    def ocl_run(self):
        self.numpy_run()

    def cuda_run(self):
        self.numpy_run()

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
        self._has_labels = len(self.original_labels) > 0
        try:
            super(FullBatchLoader, self).on_before_create_minibatch_data()
        except AttributeError:
            pass
        self._resize_validation()

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype))

    def create_originals(self, dshape):
        """
        Create original_data.mem and original_labels.mem.
        :param dshape: Future original_data.shape[1:]
        """
        self.original_data.reset(
            numpy.zeros((self.total_samples,) + dshape, self.dtype))
        self._mapped_original_labels_.reset(
            numpy.zeros(self.total_samples, Loader.LABEL_DTYPE))
        del self.original_labels[:]
        self.original_labels.extend(None for _ in range(self.total_samples))

    def fill_indices(self, start_offset, count):
        if isinstance(self.device, NumpyDevice):
            return super(FullBatchLoader, self).fill_indices(
                start_offset, count)

        self.unmap_vectors(self.original_data, self.minibatch_data,
                           self.shuffled_indices, self.minibatch_indices)

        if self.has_labels:
            self.unmap_vectors(self._mapped_original_labels_,
                               self.minibatch_labels)

        self._krn_const[0:2] = start_offset, count
        self._kernel_.set_arg(2, self._krn_const[0:1])
        self._kernel_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(self._global_size, self._local_size)

        # No further processing needed, so return True
        return True

    def fill_minibatch(self):
        for i, sample_index in enumerate(
                self.minibatch_indices.mem[:self.minibatch_size]):
            # int() is required by (guess what...) PyPy
            self.minibatch_data[i] = self.original_data[int(sample_index)]
            if self.has_labels:
                self.minibatch_labels[i] = \
                    self._mapped_original_labels_[int(sample_index)]

    def map_minibatch_labels(self):
        pass

    def analyze_dataset(self):
        """
        Override.
        """
        pass

    def normalize_minibatch(self):
        """
        Override.
        """
        pass

    def analyze_original_dataset(self):
        self.info("Normalizing to %s...", self.normalization_type)
        self.debug(
            "Data range: (%.6f, %.6f), "
            % (self.original_data.min(), self.original_data.max()))
        if self.class_lengths[TRAIN] > 0:
            self.normalizer.analyze(
                self.original_data[self.class_end_offsets[VALID]:])
        self.normalizer.normalize(self.original_data.mem)
        self.debug(
            "Normalized data range: (%.6f, %.6f), "
            % (self.original_data.min(), self.original_data.max()))

    def _resize_validation(self):
        """Extracts validation dataset from joined validation and train
        datasets randomly.

        We will rearrange indexes only.
        """
        rand = self.prng
        ratio = self.validation_ratio
        if ratio is None:
            return
        if ratio <= 0:  # Dispose validation set
            self.class_lengths[TRAIN] += self.class_lengths[VALID]
            self.class_lengths[VALID] = 0
            if self.shuffled_indices.mem is None:
                self.shuffled_indices.mem = numpy.arange(
                    self.total_samples, dtype=Loader.LABEL_DTYPE)
            return
        offs_test = self.class_lengths[TEST]
        offs = offs_test
        train_samples = self.class_lengths[VALID] + self.class_lengths[TRAIN]
        total_samples = train_samples + offs
        original_labels = self.original_labels

        if self.shuffled_indices.mem is None:
            self.shuffled_indices.mem = numpy.arange(
                total_samples, dtype=Loader.LABEL_DTYPE)
        self.shuffled_indices.map_write()
        shuffled_indices = self.shuffled_indices.mem

        # If there are no labels
        if not self.has_labels:
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
                raise ValueError(
                    "There are too few labels for class %s: %s" % (l, n_train))
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

    def _map_original_labels(self):
        self._has_labels = len(self.original_labels) > 0
        if not self.has_labels:
            return
        if len(self.labels_mapping) > 0:
            self._init_mapped_original_labels()
            return
        if len(self.original_labels) != self.original_data.shape[0]:
            raise ValueError(
                "original_labels and original_data must have the same length "
                "(%d vs %d)" % (len(self.original_labels),
                                self.original_data.shape[0]))

        for ind, lbl in enumerate(self.original_labels):
            self._samples_mapping[lbl].add(ind)

        different_labels = tuple(Counter(
            self.original_labels[i]
            for i in self.shuffled_indices[
                self.class_end_offsets[c] - self.class_lengths[c]:
                self.class_end_offsets[c]])
            for c in range(3))
        self._setup_labels_mapping(different_labels)
        self._init_mapped_original_labels()

    def _init_mapped_original_labels(self):
        self._mapped_original_labels_.reset(
            numpy.zeros(self.total_samples, Loader.LABEL_DTYPE))
        for i, label in enumerate(self.original_labels):
            self._mapped_original_labels_[i] = self.labels_mapping[label]


class FullBatchLoaderMSEMixin(LoaderMSEMixin):
    hide_from_registry = True
    """FullBatchLoader for MSE workflows.
    Attributes:
        original_targets: original target (Array).
    """
    def init_unpickled(self):
        super(FullBatchLoaderMSEMixin, self).init_unpickled()
        self._original_targets_ = memory.Array()
        self._kernel_target_ = None
        self._global_size_target = None

    @property
    def original_targets(self):
        return self._original_targets_

    def initialize(self, device, **kwargs):
        super(FullBatchLoaderMSEMixin, self).initialize(
            device=device, **kwargs)
        assert self.total_samples > 0
        self.info("Normalizing targets to %s...",
                  self.target_normalization_type)
        self.analyze_and_normalize_targets()

    def create_minibatch_data(self):
        super(FullBatchLoaderMSEMixin, self).create_minibatch_data()
        self.minibatch_targets.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.original_targets[0].shape,
            self.dtype))

    def analyze_and_normalize_targets(self):
        self.debug(
            "Target range: (%.6f, %.6f)"
            % (self.original_targets.min(), self.original_targets.max()))
        if self.class_lengths[TRAIN] > 0:
            self.target_normalizer.analyze(self.original_targets.mem)
        self.target_normalizer.normalize(self.original_targets.mem)
        if self.class_targets:
            if self.class_lengths[TRAIN] > 0:
                self.target_normalizer.analyze(self.class_targets.mem)
            self.target_normalizer.normalize(self.class_targets.mem)
        self.debug(
            "Normalized target range: (%.6f, %.6f)"
            % (self.original_targets.min(), self.original_targets.max()))

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
        self.init_vectors(self.original_targets, self.minibatch_targets)

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
        self.unmap_vectors(self.original_targets, self.minibatch_targets)
        self._kernel_target_.set_arg(2, self._krn_const[0:1])
        self._kernel_target_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(
            self._global_size_target, self._local_size_target,
            self._kernel_target_)
        return True

    def fill_minibatch(self):
        super(FullBatchLoaderMSEMixin, self).fill_minibatch()
        for i, v in enumerate(self.minibatch_indices[:self.minibatch_size]):
            # int() is required by PyPy
            self.minibatch_targets[i] = self.original_targets[int(v)]


class FullBatchLoaderMSE(FullBatchLoaderMSEMixin, FullBatchLoader):
    """FullBatchLoader for MSE workflows.
    """
    pass
