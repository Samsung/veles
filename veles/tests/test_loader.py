"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 8, 2014

Will test correctness of Loader.

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


from itertools import product
import unittest
import numpy
import os
from zope.interface import implementer

from veles.tests import AcceleratedTest, assign_backend
try:
    from veles.loader.loader_hdf5 import HDF5Loader, FullBatchHDF5Loader
    skip_hdf5 = False
except ImportError:
    HDF5Loader = FullBatchHDF5Loader = object
    skip_hdf5 = True
import veles.prng as rnd
from veles.loader import IFullBatchLoader, FullBatchLoaderMSE


@implementer(IFullBatchLoader)
class Loader(FullBatchLoaderMSE):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        N = 71599
        self.original_labels.extend([0] * N)
        self.original_data.mem = numpy.zeros([N, 28, 28],
                                             dtype=numpy.float32)
        # Will use different dtype for target
        self.original_targets.mem = numpy.zeros(
            [N, 3, 3, 3], dtype=numpy.float32)

        self.original_labels[:] = rnd.get().randint(
            0, 1000, len(self.original_labels))
        rnd.get().fill(self.original_data.mem, -100, 100)
        self.original_targets.plain[:] = rnd.get().randint(
            27, 1735, self.original_targets.size)

        self.class_lengths[0] = 0
        self.class_lengths[1] = 9737
        self.class_lengths[2] = N - self.class_lengths[1]


@assign_backend("cuda")
class TestFullBatchLoader(AcceleratedTest):
    def test_random(self):
        results = [self._test_random(*p) for p in
                   product((self.device, None), (True, False))]
        for result in results[1:]:
            for index, item in enumerate(result):
                max_diff = numpy.fabs(item - results[0][index]).max()
                self.assertLess(max_diff, 1e-6, "index = %d" % index)

    def _test_random(self, device, force_cpu, N=1000):
        rnd.get().seed(123)
        unit = Loader(self.parent, force_cpu=force_cpu, prng=rnd.get())
        unit.initialize(device, snapshot=False)
        self.assertTrue(unit.has_labels)
        res_data = numpy.zeros((N,) + unit.minibatch_data.shape,
                               dtype=unit.minibatch_data.dtype)
        res_labels = numpy.zeros((N,) + unit.minibatch_labels.shape,
                                 dtype=unit.minibatch_labels.dtype)
        res_target = numpy.zeros((N,) + unit.minibatch_targets.shape,
                                 dtype=unit.minibatch_targets.dtype)
        for i in range(N):
            unit.run()
            unit.minibatch_data.map_read()
            unit.minibatch_labels.map_read()
            unit.minibatch_targets.map_read()
            res_data[i] = unit.minibatch_data.mem
            res_labels[i] = unit.minibatch_labels.mem
            res_target[i] = unit.minibatch_targets.mem
        return res_data, res_labels, res_target


@unittest.skipIf(skip_hdf5, "h5py is unavailable")
@assign_backend("ocl")
class TestHDF5Loader(AcceleratedTest):
    def do(self, klass, **kwargs):
        csd = os.path.dirname(os.path.abspath(__file__))
        loader = klass(self.parent,
                       validation_path=os.path.join(csd, "res", "test.h5"),
                       train_path=os.path.join(csd, "res", "train.h5"))
        kwargs["snapshot"] = False
        loader.initialize(**kwargs)
        while not loader.train_ended:
            loader.run()
            loader.minibatch_data.map_read()
            loader.minibatch_labels.map_read()
            self.assertFalse((loader.minibatch_data.mem == 0).all())
            self.assertFalse((loader.minibatch_labels.mem == 0).all())

    def test_hdf5(self):
        self.do(HDF5Loader)

    def test_hdf5_fullbatch(self):
        self.do(FullBatchHDF5Loader, device=self.device)

if __name__ == "__main__":
    AcceleratedTest.main()
