# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Feb 11, 2014

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
import threading
import unittest

from veles.config import root
from veles.web_status import WebServer
from veles.tests import timeout


class Test(unittest.TestCase):

    def setUp(self):
        root.common.web.log_file = "/tmp/veles_web.test.log"
        root.common.web.port = 8071
        self.ws = WebServer()

    def tearDown(self):
        pass

    @timeout(2)
    def testStop(self):
        def stop():
            self.ws.stop()

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.ws.run()
        stopper.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testStop']
    unittest.main()
