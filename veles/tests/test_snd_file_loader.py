# -*- coding: utf-8 -*-
"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 21, 2013

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

from veles import __root__
from veles.loader.libsndfile_loader import SndFileMixin


class Test(unittest.TestCase):

    def tearDown(self):
        pass

    def testSndFileLoader(self):
        loader = SndFileMixin()
        data = loader.decode_file(
            os.path.join(__root__, "veles", "tests", "res", "sawyer.flac"))
        logging.info("%d samples at %d Hz" % (data["data"].size,
                                              data["sampling_rate"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testSndFileLoader']
    unittest.main()
