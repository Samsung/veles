# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2015

Image loaders which take data from the file system.

Copyright (c) 2015 Samsung Electronics Co., Ltd.
"""


from __future__ import division
from itertools import chain
from mimetypes import guess_type
import os
import re
import cv2
import numpy
from PIL import Image
from zope.interface import implementer, Interface

from veles.compat import from_none
import veles.error as error
from veles.loader.image import ImageLoader, IImageLoader, MODE_COLOR_MAP, \
    COLOR_CHANNELS_MAP


class IFileImageLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """


class FileImageLoaderBase(ImageLoader):
    """
    Base class for loading something from files. Function is_valid_fiename()
    should be used in child classes as filter for loading data.
    """

    def __init__(self, workflow, **kwargs):
        super(FileImageLoaderBase, self).__init__(workflow, **kwargs)
        self._filename_types = kwargs.get("filename_types", ["jpeg"])
        self._ignored_files = kwargs.get("ignored_files", [])
        self._included_files = kwargs.get("included_files", [".*"])
        self._blacklist_regexp = re.compile(
            "^%s$" % "|".join(self.ignored_files))
        self._whitelist_regexp = re.compile(
            "^%s$" % "|".join(self.included_files))

    @property
    def filename_types(self):
        return self._filename_types

    @filename_types.setter
    def filename_types(self, value):
        del self._filename_types[:]
        if isinstance(value, str):
            self._filename_types.append(value)
        else:
            self._filename_types.extend(value)

    @property
    def ignored_files(self):
        return self._ignored_files

    @ignored_files.setter
    def ignored_files(self, value):
        del self._ignored_files[:]
        if isinstance(value, str):
            self._ignored_files.append(value)
        else:
            self._ignored_files.extend(value)

    @property
    def included_files(self):
        return self._included_files

    @included_files.setter
    def included_files(self, value):
        del self._included_files[:]
        if isinstance(value, str):
            self._included_files.append(value)
        else:
            self._included_files.extend(value)

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

    def is_valid_filename(self, filename):
        """Filters the file names. Return True if the specified file path must
-        be included, otherwise, False.
        """
        if self._blacklist_regexp.match(filename):
            self.debug("Ignored %s (in black list)", filename)
            return False
        if not self._whitelist_regexp.match(filename):
            self.debug("Ignored %s (not in white list)", filename)
            return False
        mime = guess_type(filename)[0]
        if mime is None:
            self.debug("Could not determine MIME type of %s", filename)
            return False
        if not mime.startswith("image/"):
            self.debug("Ignored %s (MIME is not an image)", filename)
            return False
        mime_type_name = mime[len("image/"):]
        if mime_type_name not in self.filename_types:
            self.debug("Ignored %s (MIME %s not in the list)",
                       filename, mime_type_name)
            return False
        return True


class FileListImageLoader(FileImageLoaderBase):
    """
    Input: text file, with each line giving an image filename and label
    As with ImageLoader, it is useful for large datasets.
    """
    MAPPING = "file_list_image"

    def __init__(self, workflow, **kwargs):
        super(FileListImageLoader, self).__init__(workflow, **kwargs)
        self.path_to_test_text_file = kwargs.get("path_to_test_text_file", "")
        self.path_to_val_text_file = kwargs.get("path_to_val_text_file", "")
        self.path_to_train_text_file = kwargs.get(
            "path_to_train_text_file", "")
        self.labels = {}

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        with open(pathname, "r") as fin:
            for line in fin:
                path_to_image, _, label = line.partition(' ')
                self.labels[path_to_image] = label if label else None
                files.append(path_to_image)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return [], []
        return files

    def get_label_from_filename(self, filename):
        label = self.labels[filename]
        return label

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
class FileImageLoader(FileImageLoaderBase):
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

    def __init__(self, workflow, **kwargs):
        super(FileImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths", [])
        self.validation_paths = kwargs.get("validation_paths", [])
        self.train_paths = kwargs.get("train_paths", [])
        self.verify_interface(IFileImageLoader)

    def _check_paths(self, paths):
        if not hasattr(paths, "__iter__"):
            raise TypeError("Paths must be iterable, e.g., a list or a tuple")

    @property
    def test_paths(self):
        return self._test_paths

    @test_paths.setter
    def test_paths(self, value):
        self._check_paths(value)
        self._test_paths = value

    @property
    def validation_paths(self):
        return self._validation_paths

    @validation_paths.setter
    def validation_paths(self, value):
        self._check_paths(value)
        self._validation_paths = value

    @property
    def train_paths(self):
        return self._train_paths

    @train_paths.setter
    def train_paths(self, value):
        self._check_paths(value)
        self._train_paths = value

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        for basedir, _, filelist in os.walk(pathname):
            for name in filelist:
                full_name = os.path.join(basedir, name)
                if self.is_valid_filename(full_name):
                    files.append(full_name)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return [], []
        return files

    def get_keys(self, index):
        paths = (self.test_paths, self.validation_paths,
                 self.train_paths)[index]
        if paths is None:
            return []
        return list(
            chain.from_iterable(
                self.analyze_images(self.scan_files(p), p) for p in paths))


@implementer(IFileImageLoader)
class AutoLabelFileImageLoader(FileImageLoader):
    """
    FileImageLoader modification which takes labels by regular expression from
    file names. Unique selection groups are tracked and enumerated.
    """

    MAPPING = "auto_label_file_image"

    def __init__(self, workflow, **kwargs):
        super(AutoLabelFileImageLoader, self).__init__(workflow, **kwargs)
        # The default label is the parent directory
        self.label_regexp = re.compile(kwargs.get(
            "label_regexp", ".*%(sep)s([^%(sep)s]+)%(sep)s[^%(sep)s]+$" %
            {"sep": "\\" + os.sep}))

    def get_label_from_filename(self, filename):
        match = self.label_regexp.search(filename)
        if match is None:
            raise error.BadFormatError(
                "%s does not match label RegExp %s" %
                (filename, self.label_regexp.pattern))
        return match.group(1)
