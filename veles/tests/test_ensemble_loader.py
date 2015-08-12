#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 17, 2015

Unit tests for EnsembleLoader.

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


import logging
import os
import unittest
import numpy
from zope.interface import implementer

from veles.backends import NumpyDevice
from veles.dummy import DummyWorkflow
from veles.loader.ensemble import EnsembleLoader, IEnsembleLoader


@implementer(IEnsembleLoader)
class MyEnsembleLoader(EnsembleLoader):
    def load_winners(self):
        return [0] * 178, True


@implementer(IEnsembleLoader)
class MyEnsembleLoaderLabels(EnsembleLoader):
    def load_winners(self):
        return [0] * 178, False


class TestEnsembleLoader(unittest.TestCase):
    def test_load_data(self):
        wf = DummyWorkflow()
        loader = MyEnsembleLoader(
            wf, file=os.path.join(os.path.dirname(__file__),
                                  "res", "wine_ensemble.json"))
        loader.initialize(device=NumpyDevice())
        self.assertEqual(loader.original_data.shape, (178, 3, 3))
        self.assertTrue(loader.has_labels)
        self.assertEqual(len(loader.original_labels), 178)
        self.assertFalse(any(loader.original_labels))
        loader.run()

    def test_load_data_labels(self):
        wf = DummyWorkflow()
        loader = MyEnsembleLoaderLabels(
            wf, file=os.path.join(os.path.dirname(__file__),
                                  "res", "wine_ensemble.json"))
        loader.initialize(device=NumpyDevice())
        self.assertEqual(loader.original_data.shape, (178, 3, 3))
        self.assertTrue(loader.has_labels)
        self.assertEqual(len(loader.original_labels), 178)
        self.assertFalse(any(loader.original_labels))
        loader.run()

    def test_load_data_test(self):
        wf = DummyWorkflow()
        wf.launcher.testing = True
        loader = EnsembleLoader(
            wf, file=os.path.join(os.path.dirname(__file__),
                                  "res", "wine_ensemble.json"))
        loader.normalizer.analyze(numpy.zeros(1))
        loader.initialize(device=NumpyDevice())
        self.assertEqual(loader.original_data.shape, (178, 3, 3))
        self.assertEqual(len(loader.original_labels), 0)
        self.assertFalse(loader.has_labels)
        loader.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
