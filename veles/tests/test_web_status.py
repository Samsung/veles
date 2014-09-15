"""
Created on Feb 11, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import threading
import unittest

from veles.config import root
from veles.web_status import WebServer
from veles.tests import timeout


class Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
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
    # import sys;sys.argv = ['', 'Test.testStop']
    unittest.main()
