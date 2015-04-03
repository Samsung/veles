# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 28, 2013

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

from veles.paths import __root__


def update(root):
    root.common.update({
        "mongodb_logging_address": "smaug:27017",
        "test_dataset_root": "/data/veles",
        "help_dir": os.path.join(__root__, "docs/html"),
        "web": {
            "host": "smaug",
            "port": 8090,
        },
        "engine": {
            "source_dirs": (
                os.environ.get("VELES_ENGINE_DIRS", "").split(":") +
                [__root__])
        }
    })

    root.common.engine.device_dirs.append(os.path.join(__root__, "devices"))
    root.common.test_dataset_root = "/data/veles/datasets"
