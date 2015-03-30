"""
Created on Jan 23, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import threading
import unittest

from six import BytesIO, PY3
from twisted.internet import reactor

import veles.client as client
from veles.txzmq.connection import ZmqConnection
from veles.prng import get as get_rg
import veles.server as server
from veles.tests import DummyLauncher
from veles.workflow import Workflow


class TestWorkflow(Workflow):
    job_requested = False
    job_done = False
    update_applied = False
    power_requested = False
    job_dropped = False
    sync = threading.Event()

    def __init__(self, **kwargs):
        self.launcher = DummyLauncher()
        super(TestWorkflow, self).__init__(self.launcher, **kwargs)
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


class TestClientServer(unittest.TestCase):
    def setUp(self):
        self.master = TestWorkflow()
        self.slave = TestWorkflow()
        self.server = server.Server("127.0.0.1:5050", self.master)
        self.client = client.Client("127.0.0.1:5050", self.slave)
        self.stopper = threading.Thread(target=self.stop)
        self.stopper.start()
        self.master.thread_pool.start()

    def stop(self):
        TestWorkflow.sync.wait(1.0)
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


class TestZmqConnection(unittest.TestCase):
    def testPicklingUnpickling(self):
        class FakeSocket(object):
            def __init__(self, bio):
                self._bio = bio

            @property
            def data(self):
                return self._bio.getbuffer() if PY3 else self._bio.getvalue()

            def send(self, data, *args, **kwargs):
                self._bio.write(data)

        idata = get_rg().bytes(128000)
        bufsize = 4096
        for codec in range(4):
            socket = FakeSocket(BytesIO())
            pickler = ZmqConnection.Pickler(socket,
                                            codec if PY3 else chr(codec))
            offset = 0
            while (offset < len(idata)):
                pickler.write(idata[offset:offset + bufsize])
                offset += bufsize
            pickler.flush()
            print("Codec %d results %d bytes" % (codec, pickler.size))
            unpickler = ZmqConnection.Unpickler()
            unpickler.codec = codec if PY3 else chr(codec)
            odata = socket.data
            self.assertEqual(len(odata), pickler.size)
            offset = 0
            while (offset < len(odata)):
                unpickler.consume(odata[offset:offset + bufsize])
                offset += bufsize
            merged = unpickler.merge_chunks()
            self.assertEqual(len(idata), len(merged))
            self.assertEqual(idata, merged)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
