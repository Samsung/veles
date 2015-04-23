# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 23, 2015

Publishes results to Atlassian Confluence based on Jinja2 templates.

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


from veles.publisher.confluence import Confluence
from veles.publisher.jinja2_template_backend import Jinja2TemplateBackend


class ConfluenceBackend(Jinja2TemplateBackend):
    MAPPING = "confluence"
    TEMPLATE_EXT = "md"

    def __init__(self, template, **kwargs):
        super(ConfluenceBackend, self).__init__(template, **kwargs)
        self.confluence_access = {}
        for arg in "server", "username", "password":
            self.confluence_access[arg] = kwargs[arg]
        self.page = kwargs.get("page")
        self.parent = kwargs.get("parent")
        self.space = kwargs["space"]

    def render(self, info):
        content = super(ConfluenceBackend, self).render(info)
        page = self.page
        conf = Confluence(**self.confluence_access)
        if page is None:
            page = info["name"]
            index = 1
            while conf.get_page_summary(page, self.space) is not None:
                page = "%s (%d)" % (info["name"], index)
                index += 1
        self.info("Uploading the text (%d symbols)...", len(content))
        published = conf.store_page_content(
            page, self.space, content, parent=self.parent)
        url = published["url"]
        for name, data in info["plots"].items():
            self.info("Attaching %s...", name)
            conf.attach_file(page, self.space, {"%s.png" % name: data["png"]})
        self.info("Attaching the workflow graph...")
        conf.attach_file(page, self.space,
                         {"workflow.png": info["workflow_graph"]["png"]})
        self.info("Successfully published \"%s\" as %s", page, url)
