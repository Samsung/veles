# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 7, 2015

Interactive loaders which can be used with e.g. IPython.

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
import os
from PIL import Image
import threading
from tempfile import NamedTemporaryFile
from zope.interface import implementer
from wget import bar_adaptive, callback_progress, urllib
from veles import is_interactive

from veles.loader.base import Loader, ILoader, TEST, TRAIN, VALID
from veles.loader.image import ImageLoader, MODE_COLOR_MAP
from veles.loader.libsndfile_loader import SndFileMixin
from veles.mutable import Bool


class NotFeededError(Exception):
    pass


@implementer(ILoader)
class InteractiveLoader(Loader):
    MAPPING = "interactive"

    def __init__(self, workflow, **kwargs):
        super(InteractiveLoader, self).__init__(workflow, **kwargs)
        self._event = threading.Event()
        self._event.clear()
        self._max_minibatch_size = 1
        loader = kwargs["loader"]
        self._minibatch_data_shape = (1,) + loader.minibatch_data.shape[1:]
        self.normalization_type = loader.normalization_type
        self.normalization_parameters = loader.normalization_parameters
        self._normalizer = loader.normalizer
        self._unique_labels_count = loader.unique_labels_count
        self._labels_mapping = loader.labels_mapping
        self._loadtxt_kwargs = kwargs.get("loadtxt_kwargs", {})
        self.complete = Bool(False)
        self.derive_from(loader)

    def derive_from(self, loader):
        pass

    def reset_normalization(self):
        pass

    def load_data(self):
        assert is_interactive(), \
            "This loader may only operate in interactive mode"
        self.class_lengths[TEST] = 1
        self.class_lengths[TRAIN] = self.class_lengths[VALID] = 0

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros(self._minibatch_data_shape,
                                              dtype=self.dtype))

    def fill_minibatch(self):
        self._provide_feed()
        self.info("Waiting for the user's input...")
        try:
            self._event.wait()
        finally:
            self._revoke_feed()
            self._event.clear()

    def feed(self, obj):
        if obj is None:
            self.complete <<= True
            self._event.set()
            return
        if isinstance(obj, str):
            if os.path.exists(obj):
                obj = open(obj, "rb")
            else:
                obj = self._download(obj)
            try:
                food = self._load_from_stream(obj)
            finally:
                obj.close()
        else:
            food = obj
        if not isinstance(food, numpy.ndarray):
            raise ValueError(
                "Do not know how to digest this food type: %s", type(food))
        self._feed(food)
        self._event.set()

    def _feed(self, obj):
        self.minibatch_data.mem[0] = obj

    def _load_from_stream(self, stream):
        try:
            obj = numpy.load(stream)
            if isinstance(obj, dict):
                return next(iter(obj.values()))
            return obj
        except Exception as e:
            self.debug("Not a numpy file: %s", e)
            stream.seek(0)
        try:
            return numpy.loadtxt(stream, **self._loadtxt_kwargs)
        except Exception as e:
            self.debug("Not a text file: %s", e)
            stream.seek(0)
        try:
            img = Image.open(stream)
            self._food_color_space = MODE_COLOR_MAP[img.mode]
            return numpy.array(img)
        except Exception as e:
            self.debug("Not an image file: %s", e)
            stream.seek(0)
        try:
            with NamedTemporaryFile() as tmpf:
                tmpf.write(stream.read())
                tmpf.seek(0)
                return SndFileMixin.decode_file(tmpf.name)
        except Exception as e:
            self.debug("Not an audio file: %s", e)
            stream.seek(0)
        raise ValueError("Unknown stream type")

    @staticmethod
    def _download(url, bar_function=bar_adaptive):
        # set progress monitoring callback
        def callback_charged(blocks, block_size, total_size):
            # 'closure' to set bar drawing function in callback
            callback_progress(blocks, block_size, total_size,
                              bar_function=bar_function)

        stream = NamedTemporaryFile()
        urllib.urlretrieve(url, stream.name, callback_charged)
        print()
        stream.seek(0)
        return stream

    def _provide_feed(self):
        try:
            __IPYTHON__  # pylint: disable=E0602
            from IPython.terminal.interactiveshell import InteractiveShell
            InteractiveShell.instance().user_ns["feed"] = self.feed
        except NameError:
            globals()["feed"] = self.feed

    def _revoke_feed(self):
        try:
            __IPYTHON__  # pylint: disable=E0602
            from IPython.terminal.interactiveshell import InteractiveShell
            user_ns = InteractiveShell.instance().user_ns
            if "feed" in user_ns:
                del user_ns["feed"]
        except NameError:
            if "feed" in globals():
                del globals()["feed"]


class InteractiveImageLoader(InteractiveLoader, ImageLoader):
    MAPPING = "interactive_image"
    DISABLE_INTERFACE_VERIFICATION = True

    def derive_from(self, loader):
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
        InteractiveLoader.load_data(self)

    def create_minibatch_data(self):
        InteractiveLoader.create_minibatch_data(self)

    def fill_minibatch(self):
        InteractiveLoader.fill_minibatch(self)

    def _feed(self, data):
        color = getattr(self, "_food_color_space", self.color_space)
        bbox = ImageLoader.get_image_bbox(self, None, data.shape[:2])
        self.minibatch_data.mem[0], _, _ = self.preprocess_image(
            data, color, True, bbox)
        del self._food_color_space
