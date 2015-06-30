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

Publishes results to
`Markdown text format <http://daringfireball.net/projects/markdown/>`_.

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


import codecs
import os
import markdown
from six import string_types
import wget

from veles.external.progressbar import ProgressBar
from veles.publishing.jinja2_template_backend import Jinja2TemplateBackend


class MarkdownBackend(Jinja2TemplateBackend):
    MAPPING = "markdown"
    TEMPLATE_EXT = "md"

    def __init__(self, template, **kwargs):
        super(MarkdownBackend, self).__init__(template, **kwargs)
        self.file = kwargs.get("file")
        self.use_github_css = kwargs.get("github_css", True)
        self.html = kwargs.get("html", True)

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

    @property
    def use_github_css(self):
        return self._use_github_css

    @use_github_css.setter
    def use_github_css(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "use_github_css must be boolean (got %s)" % type(value))
        self._use_github_css = value

    @property
    def html(self):
        return self._html

    @html.setter
    def html(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "html must be boolean (got %s)" % type(value))
        self._html = value
        if value:
            self._html_template = self.environment.get_template(
                "markdown_template.html")

    def render(self, info):
        content = super(MarkdownBackend, self).render(info)
        if self.file is None:
            return content
        if isinstance(self.file, string_types):
            file = codecs.open(self.file, mode="w", encoding="utf-8",
                               errors="xmlcharrefreplace")
        else:
            file = self.file
        if not self.html:
            with file:
                file.write(content)
            return content
        with file:
            self.info("Generating HTML...")
            html = self._html_template.render(
                github_css=self.use_github_css,
                markdown=markdown.markdown(content, extensions=(
                    "markdown.extensions.smarty", "markdown.extensions.tables",
                    "markdown.extensions.codehilite",
                    "markdown.extensions.admonition", "gfm"),
                    extension_configs={"markdown.extensions.codehilite": {
                        "guess_lang": False}},
                    output_format="html5"),
                **info)
            file.write(html)
        if self.use_github_css:
            self.debug("Linked with GitHub CSS file")
        if not isinstance(self.file, string_types):
            return html
        basedir = os.path.dirname(self.file)
        fn = os.path.join(basedir, "github-markdown.css")
        if not os.path.exists(fn):
            self.info("Downloading github-markdown-css...")
            wget.download(
                "https://github.com/sindresorhus/github-markdown-css/raw/"
                "gh-pages/github-markdown.css", out=fn)
            print()
        self.info("Saving images...")
        progress = ProgressBar(2 + len(info["plots"]))
        progress.term_width = progress.maxval + 7
        progress.start()
        fn = os.path.join(basedir, "workflow.svg")
        with open(fn, "wb") as fout:
            fout.write(info["workflow_graph"]["svg"])
        progress.inc()
        self.debug("Saved %s", fn)
        fn = os.path.join(basedir, info["image"]["name"])
        with open(fn, "wb") as fout:
            fout.write(info["image"]["data"])
        progress.inc()
        self.debug("Saved %s", fn)
        for key, data in info["plots"].items():
            fn = os.path.join(basedir, "%s.svg" % key)
            with open(fn, "wb") as fout:
                fout.write(data["svg"])
            progress.inc()
            self.debug("Saved %s", fn)
        progress.finish()
        self.info("%s is ready", self.file)
        return html
