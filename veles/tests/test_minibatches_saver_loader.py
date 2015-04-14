# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 13, 2015

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


import unittest
import numpy
from zope.interface import implementer

from veles.dummy import DummyWorkflow
from veles.loader import MinibatchesSaver, MinibatchesLoader, Loader, ILoader
from veles.logger import Logger, logging


@implementer(ILoader)
class MyLoader(Loader):
    def load_data(self):
        self.counter = 0
        self.class_lengths[0] = 100 * 10
        self.class_lengths[1] = self.class_lengths[2] = 0

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros((100, 1), dtype=numpy.float32))

    def fill_minibatch(self):
        for i in range(100):
            self.minibatch_data[i] = self.counter
            self.counter += 1


class TestMinibatchesSaverLoader(unittest.TestCase, Logger):
    def setUp(self):
        self.parent = DummyWorkflow()
        self.saver = MinibatchesSaver(self.parent)
        self.loader = MinibatchesLoader(
            self.parent, shuffle_limit=0, file_name=self.saver.file_name)

    def testToTheMoonAndBack(self):
        myloader = MyLoader(self.parent, shuffle_limit=0, minibatch_size=100)
        myloader.initialize(snapshot=False)
        self.saver.link_attrs(myloader, *Loader.exports)
        self.saver.initialize(snapshot=False)
        while not myloader.epoch_ended:
            myloader.run()
            self.saver.run()
        self.saver.stop()
        self.loader.initialize(snapshot=False)
        counter = 0
        while not self.loader.epoch_ended:
            self.loader.run()
            for i in range(100):
                self.assertEqual(self.loader.minibatch_data[i], counter)
                counter += 1


if __name__ == "__main__":
    Logger.setup_logging(logging.DEBUG)
    unittest.main()
