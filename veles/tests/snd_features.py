# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 22, 2013

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
import numpy
import unittest
from veles.snd_features import SoundFeatures
from libSoundFeatureExtraction.python.sound_feature_extraction import library


class Test(unittest.TestCase):

    def setUp(self):
        library.Library(
            "/home/markhor/Development/SoundFeatureExtraction/build/src/"
            ".libs/libSoundFeatureExtraction.so")

    def tearDown(self):
        pass

    def testSoundFeatures(self):
        sf = SoundFeatures("", None)
        sf.inputs = [{"data": numpy.ones(100000, dtype=numpy.short) * 1000,
                      "sampling_rate": 16000}]
        sf.add_feature("WPP[Window(length=512, type=rectangular), DWPT, "
                       "SubbandEnergy, Log, DWPT(order=4, tree=1 2 3 3)]")
        sf.initialize()
        sf.run()
        sf.save_to_file("/tmp/test_snd_features.xml",
                        ["First and only input"])
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testSoundFeatures']
    unittest.main()
