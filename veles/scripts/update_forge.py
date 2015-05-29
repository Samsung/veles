#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Oct 20, 2014

This script update all workflows to VelesForge

Command to run: FORGE_SERVER="http://velesnet.ml/forge" PYTHONPATH=`pwd`
veles/scripts/update_forge.py

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


import logging
import subprocess
import os
import sys

from veles.config import root
from veles.scripts.generate_frontend import scan_workflows
from veles.logger import Logger


class Main(Logger):
    def run(self, server_url):
        if server_url is None:
            raise ValueError("Server URL must not be None. Looks like you did "
                             "not set FORGE_SERVER environment variable.")
        workflows = scan_workflows(False)
        for workflow in workflows:
            workflow_folder = os.path.join("veles", os.path.dirname(workflow))
            if root.common.forge.manifest in os.listdir(workflow_folder):
                self.info("Update workflow %s in VELESForge" % workflow_folder)
                subprocess.call(
                    ["python3", "-m", "veles", "forge", "upload", "-s",
                     server_url, "-d", workflow_folder])
            else:
                self.info("Workflow %s is not in Forge" % workflow)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(Main().run(os.getenv("FORGE_SERVER")))
