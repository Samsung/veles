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

Loaders which are used by RESTful API.

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

import numpy
import threading
from twisted.internet.task import LoopingCall
from zope.interface import implementer

from veles.loader.base import Loader, ILoader, TEST, TRAIN, VALID
from veles.loader.image import ImageLoader
from veles.mutable import Bool


class NotFeededError(Exception):
    pass


@implementer(ILoader)
class RestfulLoader(Loader):
    MAPPING = "restful"

    def __init__(self, workflow, **kwargs):
        super(RestfulLoader, self).__init__(workflow, **kwargs)
        self.complete = Bool(False)
        self.max_response_time = kwargs.get("max_response_time", 0.1)
        self._requests = []

    def init_unpickled(self):
        super(RestfulLoader, self).init_unpickled()
        self._event_ = threading.Event()
        self._event_.clear()
        self._lock_ = threading.Lock()
        self._flusher_ = LoopingCall(self.locked_flush)
        self._minibatch_size_ = 0

    @property
    def max_response_time(self):
        return self._max_response_time

    @max_response_time.setter
    def max_response_time(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "max_response_time must be either an integer or a floating "
                "point value (got %s)" % type(value))
        if value < 0:
            raise ValueError("max_response_time must be >= 0 (got %s)" % value)
        self._max_response_time = value

    @property
    def requests(self):
        return self._requests

    def reset_normalization(self):
        pass

    def load_data(self):
        self.class_lengths[TEST] = self.max_minibatch_size
        self.class_lengths[TRAIN] = self.class_lengths[VALID] = 0
        del self._requests[:]
        self._requests.extend((None,) * self.max_minibatch_size)
        self._flusher_.start(self.max_response_time)

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros(
            self._minibatch_data_shape, dtype=self.dtype))

    def fill_minibatch(self):
        self._minibatch_size_ = 0
        try:
            self._event_.wait()
        finally:
            self._event_.clear()

    def stop(self):
        self._flusher_.stop()
        self._event_.set()

    def feed(self, obj, request):
        assert isinstance(obj, numpy.ndarray)
        with self._lock_:
            self._feed(obj, request)
            self.requests[self._minibatch_size_] = request
            self._minibatch_size_ += 1
            if self._minibatch_size_ == self.max_minibatch_size:
                self.flush()

    def locked_flush(self):
        with self._lock_:
            self.flush()

    def flush(self):
        if self._minibatch_size_ > 0:
            self._event_.set()

    def _feed(self, obj, request):
        self.minibatch_data.mem[self._minibatch_size_] = obj


class RestfulImageLoader(RestfulLoader, ImageLoader):
    MAPPING = "restful_image"
    DISABLE_INTERFACE_VERIFICATION = True

    def derive_from(self, loader):
        super(RestfulImageLoader, self).derive_from(loader)
        self.color_space = loader.color_space
        self._original_shape = loader.original_shape
        self.path_to_mean = loader.path_to_mean
        self.add_sobel = loader.add_sobel
        self.mirror = loader.mirror
        self.scale = loader.shape
        self.scale_maintain_aspect_ratio = loader.scale_maintain_aspect_ratio
        self.rotations = loader.rotations
        self.crop = loader.crop
        self.crop_number = loader.crop_number
        self._background = loader._background
        self.background_image = loader.background_image
        self.background_color = loader.background_color
        self.smart_crop = loader.smart_crop

    def load_data(self):
        RestfulLoader.load_data(self)

    def create_minibatch_data(self):
        RestfulLoader.create_minibatch_data(self)

    def fill_minibatch(self):
        RestfulLoader.fill_minibatch(self)

    def _feed(self, data, request):
        color = request.get("color_space", self.color_space)
        bbox = ImageLoader.get_image_bbox(self, None, data.shape[:2])
        self.minibatch_data.mem[self._minibatch_size_], _, _ = \
            self.preprocess_image(data, color, True, bbox)
