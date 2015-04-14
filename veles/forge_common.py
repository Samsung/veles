# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 10, 2014

Common code between VelesForge server and client classes.

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

from pkg_resources import Requirement
from six import text_type


REQUIRED_MANIFEST_FIELDS = {
    "name", "workflow", "configuration", "short_description",
    "long_description", "author", "requires"
}


def validate_requires(requires):
    if not isinstance(requires, list):
        raise TypeError("\"requires\" must be an instance of []")
    packages = set()
    for item in requires:
        if not isinstance(item, text_type):
            raise TypeError("Each item in \"requires\" must be "
                            "a requirements.txt style string")
        pn = Requirement.parse(item).project_name
        if pn in packages:
            raise ValueError("Package %s was listed in \"requires\" more than "
                             "once" % pn)
        packages.add(pn)
