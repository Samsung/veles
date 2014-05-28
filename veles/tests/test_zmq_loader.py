"""
Created on May 28, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from twisted.internet import reactor
import threading
import unittest
import zmq

from veles.external.txzmq import ZmqConnection
from veles.pickle2 import pickle
from veles.workflow import Workflow
from veles.zmq_loader import ZeroMQLoader


class ZmqPusher(ZmqConnection):
    socketType = zmq.PUSH

    def __init__(self, *endpoints):
        super(ZmqPusher, self).__init__(endpoints)


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

    def testZmqLoader(self):
        launcher = DummyLauncher(mode=0)
        wf = Workflow(launcher)
        loader = ZeroMQLoader(wf)
        try:
            loader.initialize()
        except zmq.error.ZMQError:
            self.fail("Unable to bind")
            return
        loader.receive_data("test")
        loader.run()
        self.assertEqual("test", loader.output)
        ep = loader.generate_data_for_master()["ZmqLoaderEndpoints"]["inproc"]
        pusher = ZmqPusher(ep)
        reactor.callWhenRunning(pusher.send, pickle.dumps('hello'))

        def run():
            loader.run()
            reactor.callFromThread(reactor.stop)

        runner = threading.Thread(target=run)
        runner.start()
        reactor.run()
        runner.join()
        self.assertEqual('hello', loader.output)


if __name__ == "__main__":
    unittest.main()
