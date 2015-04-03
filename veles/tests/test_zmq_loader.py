"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 28, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import threading
import unittest

from twisted.internet import reactor
import zmq

from veles.txzmq import ZmqConnection
from veles.workflow import Workflow
from veles.zmq_loader import ZeroMQLoader
from veles.tests import timeout


class ZmqDealer(ZmqConnection):
    socketType = zmq.DEALER

    def __init__(self, *endpoints):
        super(ZmqDealer, self).__init__(endpoints)

    def request(self, cid, data):
        self.send(cid, data)


class DummyLauncher(object):
    def __init__(self, mode):
        self._mode = mode

    @property
    def is_slave(self):
        return self._mode == 0

    @property
    def is_master(self):
        return self._mode == 1

    @property
    def is_standalone(self):
        return self._mode == 2

    def add_ref(self, workflow):
        pass

    def on_workflow_finished(self):
        pass


class Test(unittest.TestCase):
    @timeout(2)
    def testZmqLoader(self):
        launcher = DummyLauncher(mode=0)
        wf = Workflow(launcher)
        loader = ZeroMQLoader(wf)
        try:
            loader.initialize()
        except zmq.error.ZMQError:
            self.fail("Unable to bind")
            return
        loader.receive_data(None, "test")
        loader.run()
        self.assertEqual("test", loader.output)
        ep = loader.generate_data_for_master()["ZmqLoaderEndpoints"]["inproc"]
        dealer = ZmqDealer(ep)
        reactor.callWhenRunning(dealer.request, b'test', b'hello')

        def run():
            loader.run()
            reactor.callFromThread(reactor.stop)

        runner = threading.Thread(target=run)
        runner.start()
        try:
            reactor.run()
        finally:
            loader.stop()
        runner.join()
        self.assertEqual(b'hello', loader.output)
        data = loader.generate_data_for_master()
        self.assertTrue("ZmqLoaderEndpoints" in data.keys())
        self.assertIsInstance(data["ZmqLoaderEndpoints"], dict)


if __name__ == "__main__":
    unittest.main()
