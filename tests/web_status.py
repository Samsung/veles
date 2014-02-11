"""
Created on Feb 11, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import time
import unittest

import web_status


class Dummy:
    def fulfill_request(self, data):
        print(str(data))
        return "AJAX WORKS!"


class Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.ws = web_status.WebStatus(Dummy())
        self.ws.run()

    def tearDown(self):
        self.ws.stop()

    def testStop(self):
        time.sleep(0.5)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testStop']
    unittest.main()
