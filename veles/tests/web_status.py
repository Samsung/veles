"""
Created on Feb 11, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import time
import unittest

import veles.config as config
from veles.web_status import WebStatus


class Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        config.web_status_log_file = "/tmp/veles_web_status_test.log"
        self.ws = WebStatus()
        self.ws.run()

    def tearDown(self):
        self.ws.stop()

    def testStop(self):
        time.sleep(0.5)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testStop']
    unittest.main()
