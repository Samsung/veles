# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 22, 2015

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
import base64

import json
import logging
from random import randint
import numpy
from six import BytesIO
from twisted.internet import reactor
from twisted.web.client import Agent, FileBodyProducer
import unittest
from twisted.web.http_headers import Headers
from zope.interface import implementer

from veles.dummy import DummyWorkflow
from veles.loader import Loader, ILoader
from veles.loader.restful import RestfulLoader
from veles.logger import Logger
from veles.memory import Array
from veles.pickle2 import pickle
from veles.plumbing import Repeater
from veles.restful_api import RESTfulAPI, NumpyJSONEncoder
from veles.tests import timeout


@implementer(ILoader)
class DummyLoader(Loader):
    def load_data(self):
        pass

    def create_minibatch_data(self):
        pass

    def fill_minibatch(self):
        pass


class RESTAPITest(unittest.TestCase):
    @timeout()
    def test_workflow(self):
        workflow = DummyWorkflow()
        workflow.run_is_blocking = False
        repeater = Repeater(workflow)
        repeater.link_from(workflow.start_point)
        port = 6565 + randint(-1000, 1000)
        api = RESTfulAPI(workflow, port=port, path="/api")
        api.link_from(repeater)
        base_loader = DummyLoader(workflow)
        base_loader.minibatch_data = numpy.zeros((10, 10, 10))
        loader = RestfulLoader(workflow, loader=base_loader, minibatch_size=1)
        loader.link_from(api)
        workflow.del_ref(base_loader)
        api.link_attrs(loader, "feed", "requests", "minibatch_size")
        api.results = [numpy.ones((3, 3))]
        repeater.link_from(loader)
        workflow.end_point.link_from(api).unlink_from(workflow.start_point)
        workflow.end_point.gate_block <<= True
        loader.gate_block = ~workflow.end_point.gate_block
        workflow.initialize(snapshot=False)

        run = api.run

        def finish_run():
            workflow.end_point.gate_block <<= not api.run_was_called
            run()

        api.run = finish_run
        reactor.callWhenRunning(workflow.run)

        agent = Agent(reactor)
        headers = Headers({b'User-Agent': [b'twisted'],
                           b'Content-Type': [b'application/json']})
        body = FileBodyProducer(BytesIO(json.dumps(
            {"input": numpy.ones((10, 10)).tolist(),
             "codec": "list"}).encode('charmap')))
        d = agent.request(
            b'POST', ("http://localhost:%d/api" % port).encode('charmap'),
            headers=headers, bodyProducer=body)
        response = [None]

        def finished(result):
            print("Received the result %d" % result.code)
            response[0] = result
            reactor.stop()

        def failed(error):
            error.printTraceback()
            reactor.stop()
            self.fail()

        d.addCallback(finished)
        d.addErrback(failed)
        reactor.run()
        self.assertIsNotNone(response[0])
        self.assertEqual(response[0].code, 200)
        # We should use deliverBody here, but the response is small enough
        self.assertEqual(
            response[0]._bodyBuffer[0],
            b'{"result": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}')

    def test_pickling(self):
        workflow = DummyWorkflow()
        port = 6565 + randint(-1000, 1000)
        api = RESTfulAPI(workflow, port=port)
        base_loader = DummyLoader(workflow)
        base_loader.minibatch_data = numpy.zeros((10, 10, 10))
        loader = RestfulLoader(workflow, loader=base_loader, minibatch_size=1)
        workflow.del_ref(base_loader)
        api.link_attrs(loader, "feed", "requests", "minibatch_size")
        api.results = [numpy.ones((3, 3))]
        api.initialize(snapshot=False)
        new_api = pickle.loads(pickle.dumps(api))
        self.assertIsInstance(new_api.results, list)
        self.assertEqual(len(new_api.results), 1)
        self.assertTrue((new_api.results[0] == numpy.ones((3, 3))).all())
        self.assertIsInstance(new_api.requests, list)
        self.assertEqual(len(new_api.requests), 0)

    def test_map_read(self):
        vec = Array(numpy.ones((10, 10)))
        arr = json.loads(json.dumps(vec, cls=NumpyJSONEncoder))
        self.assertIsInstance(arr, list)
        nparr = numpy.array(arr)
        self.assertEqual(nparr.shape, (10, 10))
        self.assertTrue((nparr == 1).all())

    def test_decode_base64(self):
        arr = numpy.ones((10, 20), dtype=numpy.int32)
        fake_resp = {"shape": list(arr.shape), "type": "int32="}
        data = RESTfulAPI._decode_base64.__get__(self)(
            None, fake_resp, base64.b64encode(arr.data))
        self.assertTrue((arr == data).all())
        self.assertEqual(arr.shape, data.shape)
        self.assertEqual(arr.dtype, data.dtype)

if __name__ == "__main__":
    Logger.setup_logging(logging.DEBUG)
    unittest.main()
