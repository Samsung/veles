#!/usr/bin/python3

"""
Created on Feb 14, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import socket
import threading
from twisted.internet import reactor
import unittest

from veles.config import root
from veles.launcher import Launcher, filter_argv
from veles.workflow import Workflow


class TestWorkflow(Workflow):
    job_requested = False
    job_done = False
    job_dropped = False
    update_applied = False
    power_requested = False
    event = threading.Event()

    def __init__(self, launcher, **kwargs):
        super(TestWorkflow, self).__init__(launcher, **kwargs)

    @Workflow.run_timed
    @Workflow.method_timed
    def generate_data_for_slave(self, slave):
        TestWorkflow.job_requested = True
        return [{'objective': 'win'}]

    @Workflow.run_timed
    @Workflow.method_timed
    def apply_data_from_slave(self, update, slave):
        if TestWorkflow.update_applied:
            TestWorkflow.event.set()
        if isinstance(update, list) and isinstance(update[0], dict) and \
                update[0]['objective'] == 'win':
            TestWorkflow.update_applied = True
            return True
        return False

    def do_job(self, job, update, callback):
        if isinstance(job, list) and isinstance(job[0], dict) and \
                job[0]['objective'] == 'win':
            TestWorkflow.job_done = True
        callback(job)

    def drop_slave(self, slave):
        TestWorkflow.job_dropped = True

    @property
    def computing_power(self):
        TestWorkflow.power_requested = True
        return 100


class TestLauncher(unittest.TestCase):

    def setUp(self):
        root.common.web.host = socket.gethostname()
        self.server = Launcher(listen_address="localhost:9999",
                               web_status=False)
        self.client = Launcher(master_address="localhost:9999")
        self.master_workflow = TestWorkflow(self.server)
        self.slave_workflow = TestWorkflow(self.client)

    def tearDown(self):
        pass

    def testConnectivity(self):
        reactor.callLater(0.1, reactor.stop)
        self.stopper = threading.Thread(target=self.stop)
        self.stopper.start()
        self.server.run()
        self.stopper.join()
        self.assertTrue(TestWorkflow.job_requested, "Job was not requested.")
        self.assertTrue(TestWorkflow.job_done, "Job was not done.")
        self.assertTrue(TestWorkflow.update_applied, "Update was not applied.")
        self.assertTrue(TestWorkflow.power_requested,
                        "Power was not requested.")

    def stop(self):
        TestWorkflow.event.wait(0.1)
        reactor.callFromThread(reactor.stop)


class TestGlobal(unittest.TestCase):
    def testFilterArgv(self):
        argv = ["-v", "--listen", "0.0.0.0:5000", "-p", "-k=3000", "-e",
                "--full", "kwarg", "other", "-x", "--rec"]
        f = filter_argv(argv, "--listen", "-k", "--full", "-x")
        self.assertEqual(f, ["-v", "-p", "-e", "other", "--rec"],
                         "filter_argv failed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
