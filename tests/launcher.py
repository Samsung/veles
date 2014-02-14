#!/usr/bin/python3

"""
Created on Feb 14, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


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
    update_applied = False
    power_requested = False

    def __init__(self, workflow, **kwargs):
        super(TestWorkflow, self).__init__(workflow, **kwargs)

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

    def setUp(self):
        self.master_workflow = TestWorkflow(None)
        self.slave_workflow = TestWorkflow(None)
        veles.config.web_status_host = socket.gethostname()
        self.server = Launcher(self.master_workflow, mode="master",
                               addr="localhost:9999")
        self.client = Launcher(self.slave_workflow, mode="slave",
                               addr="localhost:9999")

    def tearDown(self):
        pass

    def testConnectivity(self):
        reactor.callLater(0.1, reactor.stop)
        reactor.run()
        time.sleep(1)
        reactor.stop()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testConnectivity']
    unittest.main()
