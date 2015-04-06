# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mar 30, 2015

Base classes which take data from the file system

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
from mimetypes import guess_type
import os
import re
from zope.interface import Interface, implementer

import veles.error as error
from veles.units import Unit


class IFileLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """


class FileFilter(Unit):
    hide_from_registry = True
    """
    Base class for loading something from files. Function is_valid_fiename()
    should be used in child classes as filter for loading data.
    """

    def __init__(self, workflow, **kwargs):
        super(FileFilter, self).__init__(workflow, **kwargs)
        self._ignored_files = kwargs.get("ignored_files", [])
        self._included_files = kwargs.get("included_files", [".*"])
        self._file_type = kwargs["file_type"]
        self._file_subtypes = kwargs["file_subtypes"]
        self.update_filters()

    def update_filters(self):
        self._update_whitelist_filter()
        self._update_blacklist_filter()
        self._update_mime()

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
        self._update_blacklist_filter()

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
        self._update_whitelist_filter()

    @property
    def mime(self):
        return self._mime

    @property
    def file_type(self):
        return self._file_type

    @property
    def file_subtypes(self):
        return self._file_subtypes

    @file_subtypes.setter
    def file_subtypes(self, value):
        del self._file_subtypes[:]
        if isinstance(value, str):
            self._file_subtypes.append(value)
        else:
            self._file_subtypes.extend(value)
        self._update_mime()

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
        return self._mime_regexp.match(mime) is not None

    def _update_blacklist_filter(self):
        self._blacklist_regexp = re.compile(
            "^%s$" % "|".join(self.ignored_files))

    def _update_whitelist_filter(self):
        self._whitelist_regexp = re.compile(
            "^%s$" % "|".join(self.included_files))

    def _update_mime(self):
        self._mime = "%s/(%s)" % (
            self._file_type, "|".join(self._file_subtypes))
        self._mime_regexp = re.compile(self._mime)


@implementer(IFileLoader)
class FileListLoaderBase(Unit):
    hide_from_registry = True

    def __init__(self, workflow, **kwargs):
        super(FileListLoaderBase, self).__init__(workflow, **kwargs)
        self.path_to_test_text_file = kwargs.get("path_to_test_text_file", "")
        self.path_to_val_text_file = kwargs.get("path_to_val_text_file", "")
        self.path_to_train_text_file = kwargs["path_to_train_text_file"]
        self._labels = {}

    @property
    def labels(self):
        return self._labels

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        if pathname.endswith(".json"):
            with open(pathname, "r") as fin:
                images = json.load(fin)
                for image in images.values():
                    if len(image["label"]) > 0:
                        label = image["label"][0]
                        self.labels[image["path"]] = label
                        files.append(image["path"])
        else:
            with open(pathname, "r") as fin:
                for line in fin:
                    path_to_image, _, label = line.partition(' ')
                    if label:
                        self.labels[path_to_image] = label
                        files.append(path_to_image)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return []
        return files

    def get_label_from_filename(self, filename):
        return self.labels[filename]


class FileLoaderBase(FileFilter):
    hide_from_registry = True
    """Provides methods to load data from multiple folders.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].

    Must be overriden in child classes:
        get_label_from_filename()
        is_valid_filename()
    """

    def __init__(self, workflow, **kwargs):
        super(FileLoaderBase, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths", [])
        self.validation_paths = kwargs.get("validation_paths", [])
        self.train_paths = kwargs.get("train_paths", [])
        self.verify_interface(IFileLoader)

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


@implementer(IFileLoader)
class AutoLabelFileLoader(FileLoaderBase):
    hide_from_registry = True
    """
    FileLoader extension which takes labels by regular expression from
    file names. Unique selection groups are tracked and enumerated.
    """

    def __init__(self, workflow, **kwargs):
        super(AutoLabelFileLoader, self).__init__(workflow, **kwargs)
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
