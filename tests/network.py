"""
Created on Jan 23, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import pickle
from time import sleep
from twisted.internet import reactor
import unittest

import client
import network_common
import server
import workflow


class TestWorkflow(workflow.Workflow):
    job_requested = False
    job_done = False
    update_applied = False
    power_requested = False

    def __init__(self):
        super(TestWorkflow, self).__init__()

    def request_job(self):
        TestWorkflow.job_requested = True
        return pickle.dumps({'objective': 'win'})

    def do_job(self, data):
        job = pickle.loads(data)
        if isinstance(job, dict):
            TestWorkflow.job_done = True
        return data

    def apply_update(self, update):
        obj = pickle.loads(update)
        if isinstance(obj, dict):
            TestWorkflow.update_applied = True
            return True
        return False

    def get_computing_power(self):
        TestWorkflow.power_requested = True
        return 100


class Test(unittest.TestCase):
    CONFIG_FILE_NAME = "/tmp/test_network_veles_conf.py"

    def setUp(self):
        conf = open(Test.CONFIG_FILE_NAME, "w")
        conf.write("{'%s': '127.0.0.1', '%s': 5050}" %
                   (network_common.NetworkConfigurable.CONFIG_ADDRESS,
                    network_common.NetworkConfigurable.CONFIG_PORT))
        conf.close()
        self.master = TestWorkflow()
        self.slave = TestWorkflow()
        self.server = server.Server(Test.CONFIG_FILE_NAME, self.master)
        self.client = client.Client(Test.CONFIG_FILE_NAME, self.slave)
        reactor.callLater(0.1, reactor.stop)

    def tearDown(self):
        pass

    def testWork(self):
        self.server.run()
        self.assertTrue(TestWorkflow.job_requested, "Job was not requested.")
        self.assertTrue(TestWorkflow.job_done, "Job was not done.")
        self.assertTrue(TestWorkflow.update_applied, "Update was not applied.")
        self.assertTrue(TestWorkflow.power_requested,
                        "Power was not requested.")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testConnection']
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
