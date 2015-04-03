# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Feb 26, 2015

Image loaders which take data from the file system.

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


from __future__ import division
from itertools import chain
import cv2
import numpy
from PIL import Image
from zope.interface import implementer

from veles.compat import from_none
import veles.error as error
from veles.loader.file_loader import AutoLabelFileLoader, FileFilter, \
    FileLoaderBase, FileListLoaderBase
from veles.loader.image import ImageLoader, IImageLoader, MODE_COLOR_MAP, \
    COLOR_CHANNELS_MAP


class FileImageLoaderBase(ImageLoader, FileFilter):
    """
    Base class for loading something from files. Function is_valid_fiename()
    should be used in child classes as filter for loading data.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["file_type"] = "image"
        kwargs["file_subtypes"] = kwargs.get("file_subtypes", ["jpeg", "png"])
        super(FileImageLoaderBase, self).__init__(workflow, **kwargs)

    def get_image_info(self, key):
        """
        :param key: The full path to the analysed image.
        :return: tuple (image size, number of channels).
        """
        try:
            with open(key, "rb") as fin:
                img = Image.open(fin)
                return tuple(reversed(img.size)), MODE_COLOR_MAP[img.mode]
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
            # Unable to read the image with PIL. Fall back to slow OpenCV
            # method which reads the whole image.
            img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return img.shape[:2], "BGR"

    def get_image_data(self, key):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        try:
            with open(key, "rb") as fin:
                img = Image.open(fin)
                if img.mode in ("P", "CMYK"):
                    return numpy.array(img.convert("RGB"),
                                       dtype=self.source_dtype)
                else:
                    return numpy.array(img, dtype=self.source_dtype)
        except (TypeError, KeyboardInterrupt) as e:
            raise from_none(e)
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
            img = cv2.imread(key)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return img.astype(self.source_dtype)

    def get_image_label(self, key):
        return self.get_label_from_filename(key)

    def analyze_images(self, files, pathname):
        # First pass: get the final list of files and shape
        self.debug("Analyzing %d images in %s", len(files), pathname)
        uniform_files = []
        for file in files:
            size, color_space = self.get_image_info(file)
            shape = size + (COLOR_CHANNELS_MAP[color_space],)
            if (not isinstance(self.scale, tuple) and
                    self.uncropped_shape != tuple() and
                    shape[:2] != self.uncropped_shape):
                self.warning("%s has the different shape %s (expected %s)",
                             file, shape[:2], self.uncropped_shape)
            else:
                if self.uncropped_shape == tuple():
                    self.original_shape = shape
                uniform_files.append(file)
        return uniform_files


@implementer(IImageLoader)
class FileListImageLoader(FileImageLoaderBase, FileListLoaderBase):
    """
    Input: text file, with each line giving an image filename and label
    As with ImageLoader, it is useful for large datasets.
    """
    MAPPING = "file_list_image"

    def get_keys(self, index):
        paths = (
            self.path_to_test_text_file,
            self.path_to_val_text_file,
            self.path_to_train_text_file)[index]
        if paths is None:
            return []
        return list(
            chain.from_iterable(
                self.analyze_images(self.scan_files(p), p) for p in paths))


@implementer(IImageLoader)
class FileImageLoader(FileImageLoaderBase, FileLoaderBase):
    """Loads images from multiple folders. As with ImageLoader, it is useful
    for large datasets.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].

    Must be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """

    def get_keys(self, index):
        paths = (self.test_paths, self.validation_paths,
                 self.train_paths)[index]
        if paths is None:
            return []
        return list(
            chain.from_iterable(
                self.analyze_images(self.scan_files(p), p) for p in paths))


class AutoLabelFileImageLoader(FileImageLoader, AutoLabelFileLoader):
    """
    FileImageLoader extension which takes labels by regular expression from
    file names. Unique selection groups are tracked and enumerated.
    """

    MAPPING = "auto_label_file_image"
