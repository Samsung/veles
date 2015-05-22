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

RESTful API unit.

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

import json
from itertools import islice
from time import strftime, localtime
import numpy
from twisted.internet import reactor
from twisted.web.server import Site, NOT_DONE_YET
from twisted.web.resource import Resource, NoResource
from zope.interface import implementer

from veles.config import root
from veles.distributable import TriviallyDistributable, IDistributable
from veles.units import IUnit, Unit


class APIResource(Resource):
    isLeaf = True

    def __init__(self, path, callback):
        Resource.__init__(self)
        self._path = path.encode('charmap')
        self._callback = callback

    def render_POST(self, request):
        if request.path != self._path:
            page = NoResource(
                message="API path %s is not supported" % request.URLPath())
            return page.render(request)
        if request.getHeader(b"Content-Type") != b"application/json":
            page = NoResource(
                message="Unsupported Content-Type (must be \"application/json"
                        "\")" % request.URLPath())
            return page.render(request)
        request.setHeader(b"User-Agent", [b"twisted"])
        request.setHeader(b"Content-Type", [b"application/json"])
        self._callback(request)
        return NOT_DONE_YET


@implementer(IUnit, IDistributable)
class RESTfulAPI(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "SERVICE"
        super(RESTfulAPI, self).__init__(workflow, **kwargs)
        self.port = kwargs.get("port", root.common.api.port)
        self.path = kwargs.get("path", root.common.api.path)
        self.demand("feed", "requests", "results", "minibatch_size")

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        if not isinstance(value, int):
            raise ValueError("port must be an integer (got %s)" % type(value))
        if value < 1 or value > 65535:
            raise ValueError("port is out of range (%d)" % value)
        self._port = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not value.startswith('/'):
            raise ValueError("Invalid path: %s", value)
        self._path = value

    def initialize(self, **kwargs):
        reactor.listenTCP(self.port, Site(APIResource(self.path, self.serve)))
        self.info("Listening on 0.0.0.0:%d%s", self.port, self.path)

    def run(self):
        reactor.callFromThread(self.respond)

    def respond(self):
        for request, result in islice(zip(self.requests, self.results),
                                      0, self.minibatch_size):
            if request is None:
                continue
            request.write(json.dumps({"result": result}).encode("utf-8"))
            request.finish()

    def fail(self, request, message):
        self.warning(message)
        request.setResponseCode(400)
        request.write(json.dumps({"error": message}).encode('utf-8'))
        request.finish()

    def serve(self, request):
        raw_response = request.content.read()
        try:
            response = json.loads(raw_response.decode('utf-8'))
        except ValueError:
            self.fail(request, "Failed to parse JSON")
            return
        if not isinstance(response, dict) or "input" not in response or \
                not isinstance(response["input"], list):
            self.fail(request, "Invalid input format")
            return
        try:
            data = numpy.array(response["input"], numpy.float32)
        except ValueError:
            self.fail(request, "Invalid input array format")
            return
        try:
            self.feed(data, request)
        except Exception as e:
            self.fail(request, "Invalid input value: %s" % e)
        self.debug("%s: received %d bytes", strftime("%X", localtime()),
                   len(raw_response))
