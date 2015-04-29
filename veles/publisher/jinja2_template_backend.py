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

Base class for publishing backends which use Jinja2 templates.

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
import sys
from jinja2 import Environment, FileSystemLoader

from veles.publisher.backend import Backend


class Jinja2TemplateBackend(Backend):
    def __init__(self, template, **kwargs):
        super(Jinja2TemplateBackend, self).__init__(template, **kwargs)
        self.environment = Environment(
            trim_blocks=kwargs.get("trim_blocks", True),
            lstrip_blocks=kwargs.get("lstrip_blocks", True),
            loader=FileSystemLoader(os.path.dirname(
                sys.modules[type(self).__module__].__file__)))
        if template is None:
            self.template = self.environment.get_template(
                "%s_template.%s" % (self.MAPPING, self.TEMPLATE_EXT))
        else:
            self.template = self.environment.from_string(template)

    def render(self, info):
        self.info("Rendering the template...")
        content = self.template.render(**info)
        self.debug("Rendered:\n%s", content)
        return content
