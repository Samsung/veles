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

Created on June 25, 2015

Unit tests for :class:`veles.downloader.Downloader`.

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
import random
from six import PY3
from subprocess import Popen
import sys
from tempfile import gettempdir
from time import sleep
import unittest

from veles.downloader import Downloader
from veles.dummy import DummyWorkflow


class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.parent = DummyWorkflow()

    def test_download(self):
        random.seed()
        module = "http.server" if PY3 else "SimpleHTTPServer"
        port = random.randint(2048, 32000)
        with open(os.devnull, "w") as devnull:
            # Python 2.7 subprocess package does not have DEVNULL
            server = Popen((sys.executable, "-m", module, str(port)),
                           cwd=os.path.join(os.path.dirname(__file__), "res"),
                           stderr=devnull)
        sleep(1)
        tempdir = gettempdir()
        file = "wine_ensemble.json"
        if not hasattr(sys.stdout, "fileno"):
            sys.stdout.fileno = lambda: 0
        try:
            downloader = Downloader(
                self.parent,
                url="http://localhost:%d/wine_ensemble.json.tar.gz" % port,
                files=(file,), directory=tempdir)
            downloader.initialize()
        finally:
            server.terminate()
        path = os.path.join(tempdir, file)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.getsize(path), 36852)
        os.remove(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
