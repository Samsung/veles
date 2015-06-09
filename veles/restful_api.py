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
import base64

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
from veles.memory import Array
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
        request.setHeader(b"Content-Type", b"application/json")
        self._callback(request)
        return NOT_DONE_YET


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Array):
            obj.map_read()
            obj = obj.mem
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.number):
            return obj.item()
        elif isinstance(obj, (complex, numpy.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode("charmap")
        return json.JSONEncoder.default(self, obj)


@implementer(IUnit, IDistributable)
class RESTfulAPI(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "SERVICE"
        super(RESTfulAPI, self).__init__(workflow, **kwargs)
        self.port = kwargs.get("port", root.common.api.port)
        self.path = kwargs.get("path", root.common.api.path)
        self.demand("feed", "requests", "results", "minibatch_size")

    def init_unpickled(self):
        super(RESTfulAPI, self).init_unpickled()
        self._listener_ = None

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
        self._listener_ = reactor.listenTCP(
            self.port, Site(APIResource(self.path, self.serve)))
        self.info("Listening on 0.0.0.0:%d%s", self.port, self.path)

    def run(self):
        reactor.callFromThread(self.respond)

    def stop(self):
        if self._listener_ is not None:
            self._listener_.stopListening()

    def respond(self):
        for request, result in islice(zip(self.requests, self.results),
                                      0, self.minibatch_size):
            if request is None:
                continue
            request.write(json.dumps({"result": result},
                                     cls=NumpyJSONEncoder).encode("utf-8"))
            request.finish()

    def fail(self, request, message):
        self.warning(message)
        request.setResponseCode(400)
        request.write(json.dumps({"error": message}).encode('utf-8'))
        request.finish()

    def _decode_base64(self, request, response, input_obj):
        # base64 codec
        if "shape" not in response:
            self.fail(request, "There is no \"shape\" attribute which "
                               "defines the input array shape")
            return None
        shape = response["shape"]
        if not isinstance(shape, list) or len(shape) < 1:
            self.fail(request, "\"shape\" must be a non-trivial array")
            return None
        if "type" not in response:
            self.fail(request, "There is no \"type\" attribute which "
                               "defines the array data type (e.g., "
                               "\"float32\" or \"uint8\", see numpy.dtype)"
                               ".")
            return None
        dtype_name = response["type"]
        if dtype_name is None:
            # this will result in numpy.float64
            self.fail(request, "\"type\" must not be null")
            return None
        if dtype_name[-1] in "<=>":
            byte_order = dtype_name[-1]
            dtype_name = dtype_name[:-1]
        else:
            byte_order = None
        try:
            dtype = numpy.dtype(dtype_name)
        except TypeError:
            self.fail(request, "Invalid \"type\" value. For the list of "
                               "supported values, see numpy.dtype.")
            return None
        if byte_order is not None:
            dtype = dtype.newbyteorder(byte_order)
        try:
            buffer = base64.b64decode(input_obj)
        except base64.binascii.Error as e:
            self.fail(request, "Failed to decode base64: %s." % e)
            return None
        try:
            return numpy.frombuffer(buffer, dtype).reshape(shape)
        except Exception as e:
            self.fail(request, "Failed to create the numpy array: %s." % e)
            return None

    def serve(self, request):
        raw_response = request.content.read()
        try:
            response = json.loads(raw_response.decode('utf-8'))
        except ValueError:
            self.fail(request, "Failed to parse JSON")
            return
        if not isinstance(response, dict) or "input" not in response \
                or "codec" not in response:
            self.fail(request, "Invalid input format: there must be \"input\" "
                               "and \"codec\" attributes")
            return
        input_obj = response["input"]
        codec = response["codec"]
        if codec not in ("list", "base64"):
            self.fail(request, "Invalid codec value: must be either \"list\" "
                               "or \"base64\"")
            return
        if codec == "list":
            try:
                data = numpy.array(input_obj, numpy.float32)
            except ValueError:
                self.fail(request, "Invalid input array format")
                return
        else:
            data = self._decode_base64(request, response, input_obj)
            if data is None:
                return
        try:
            self.feed(data, request)
        except Exception as e:
            self.fail(request, "Invalid input value: %s" % e)
        self.debug("%s: received %d bytes", strftime("%X", localtime()),
                   len(raw_response))
