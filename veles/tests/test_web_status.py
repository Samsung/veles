"""
Created on Feb 11, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import threading
import unittest

from veles.config import root
from veles.web_status import WebStatus
from veles.tests import timeout


class Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        root.common.web_status_log_file = "/tmp/veles_web_status_test.log"
        self.ws = WebStatus()

    def tearDown(self):
        pass

    @timeout(2)
    def testStop(self):
        def stop():
            self.ws.running.wait()
            self.ws.stop()

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.ws.run()
        stopper.join()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testStop']
    unittest.main()
