# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 30, 2015

Extension for MarkdownBackend to write PDF reports.

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


import os
from tempfile import mkdtemp
from shutil import rmtree
from six import string_types
from weasyprint import HTML

from veles.publishing.backend import Backend
from veles.publishing.markdown_backend import MarkdownBackend


class PDFBackend(Backend):
    MAPPING = "pdf"

    def __init__(self, template, **kwargs):
        super(PDFBackend, self).__init__(template, **kwargs)
        self.markdown = MarkdownBackend(
            template, html=True, file=os.path.join(mkdtemp(
                prefix="veles-pdf-publish-"), "report.html"),
            image_format="png")
        self.file = kwargs["file"]

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        if value is None:
            self._file = None
            return
        if not isinstance(value, string_types) and not hasattr(value, "write"):
            raise TypeError(
                "file must either a path or a file-like object (got %s)" %
                type(value))
        self._file = value

    def render(self, info):
        try:
            self.markdown.render(info)
            HTML(self.markdown.file).write_pdf(self.file)
            self.info("%s is ready", self.file)
        finally:
            rmtree(os.path.dirname(self.markdown.file))
