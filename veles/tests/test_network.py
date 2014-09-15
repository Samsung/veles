"""
Created on Jan 23, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import threading
from twisted.internet import reactor
import unittest

import veles.client as client
import veles.server as server
from veles.workflow import Workflow
from veles.tests import DummyLauncher


class TestWorkflow(Workflow):
    job_requested = False
    job_done = False
    update_applied = False
    power_requested = False
    job_dropped = False
    sync = threading.Event()

    def __init__(self, **kwargs):
        super(TestWorkflow, self).__init__(DummyLauncher(), **kwargs)
        self.is_running = True

    @Workflow.run_timed
    @Workflow.method_timed
    def generate_data_for_slave(self, slave):
        TestWorkflow.job_requested = True
        return {'objective': 'win'}

    def do_job(self, job, update, callback):
        if isinstance(job, dict):
            TestWorkflow.job_done = True
        callback(job)

    @Workflow.run_timed
    @Workflow.method_timed
    def apply_data_from_slave(self, obj, slave):
        if TestWorkflow.update_applied:
            TestWorkflow.sync.set()
        if isinstance(obj, dict):
            TestWorkflow.update_applied = True
            return True
        return False

    def drop_slave(self, slave):
        TestWorkflow.job_dropped = True

    @property
    def computing_power(self):
        TestWorkflow.power_requested = True
        return 100

    @property
    def is_slave(self):
        return False

    @property
    def is_master(self):
        return False

    @property
    def is_standalone(self):
        return True

    def add_ref(self, workflow):
        pass


class Test(unittest.TestCase):
    def setUp(self):
        self.master = TestWorkflow()
        self.slave = TestWorkflow()
        self.server = server.Server("127.0.0.1:5050", self.master)
        self.client = client.Client("127.0.0.1:5050", self.slave)
        self.stopper = threading.Thread(target=self.stop)
        self.stopper.start()

    def stop(self):
        TestWorkflow.sync.wait(0.1)
        reactor.callFromThread(reactor.stop)

    def tearDown(self):
        pass

    def testWork(self):
        reactor.run()
        self.stopper.join()
        self.assertTrue(TestWorkflow.job_requested, "Job was not requested.")
        self.assertTrue(TestWorkflow.job_done, "Job was not done.")
        self.assertTrue(TestWorkflow.update_applied, "Update was not applied.")
        self.assertTrue(TestWorkflow.power_requested,
                        "Power was not requested.")
        self.assertTrue(TestWorkflow.job_dropped,
                        "Job was not dropped in the end.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
