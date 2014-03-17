#!/usr/bin/python3

"""
Created on Feb 14, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import pickle
import socket
import time
from twisted.internet import reactor
import unittest

import veles.config
from veles.launcher import Launcher
import veles.workflows


class TestWorkflow(veles.workflows.Workflow):
    job_requested = False
    job_done = False
    job_dropped = False
    update_applied = False
    power_requested = False

    def __init__(self, workflow, **kwargs):
        super(TestWorkflow, self).__init__(workflow, **kwargs)

    def request_job(self, slave):
        TestWorkflow.job_requested = True
        return pickle.dumps({'objective': 'win'})

    def do_job(self, data):
        job = pickle.loads(data)
        if isinstance(job, dict):
            TestWorkflow.job_done = True
        return data

    def apply_update(self, update, slave):
        obj = pickle.loads(update)
        if isinstance(obj, dict):
            TestWorkflow.update_applied = True
            return True
        return False

    def drop_slave(self, slave):
        TestWorkflow.job_dropped = True

    def get_computing_power(self):
        TestWorkflow.power_requested = True
        return 100


class Test(unittest.TestCase):

    def setUp(self):
        self.master_workflow = TestWorkflow(None)
        self.slave_workflow = TestWorkflow(None)
        veles.config.web_status_host = socket.gethostname()
        self.server = Launcher(listen_address="localhost:9999",
                               web_status=False)
        self.server.initialize(self.master_workflow)
        self.client = Launcher(master_address="localhost:9999")
        self.client.initialize(self.slave_workflow)

    def tearDown(self):
        pass

    def testConnectivity(self):
        reactor.callLater(0.1, self.server.stop)
        self.server.run()
        self.assertTrue(TestWorkflow.job_requested, "Job was not requested.")
        self.assertTrue(TestWorkflow.job_done, "Job was not done.")
        self.assertTrue(TestWorkflow.job_dropped,
                        "Job was not dropped in the end.")
        self.assertTrue(TestWorkflow.update_applied, "Update was not applied.")
        self.assertTrue(TestWorkflow.power_requested,
                        "Power was not requested.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
