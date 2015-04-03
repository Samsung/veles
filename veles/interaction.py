# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 7, 2014

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


from IPython.config.loader import Config
from IPython.terminal.embed import InteractiveShellEmbed
import select
import sys
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
from veles.units import Unit, IUnit


@implementer(IUnit)
class Shell(Unit, TriviallyDistributable):
    """
    Runs embedded IPython
    """
    BANNER1 = "\nVELES interactive console"
    BANNER2 = "Type in 'workflow' or 'units' to start"

    def __init__(self, workflow, **kwargs):
        super(Shell, self).__init__(workflow, **kwargs)
        self.cfg = Config()
        self.cfg.PromptManager.in_template = "veles [\\#]> "
        self.cfg.PromptManager.out_template = "veles [\\#]: "
        self.cfg.HistoryManager.enabled = False

    def initialize(self, **kwargs):
        if self.interactive:
            return
        self.shell_ = InteractiveShellEmbed(config=self.cfg,
                                            banner1=Shell.BANNER1,
                                            banner2=Shell.BANNER2)

    def interact(self, extra_locals=None):
        workflow = self.workflow  # pylint: disable=W0612
        units = self.workflow.units  # pylint: disable=W0612
        exec('\n'.join("%s = extra_locals['%s']" % (k, k)
                       for k in extra_locals or {}))
        self.shell_()

    @staticmethod
    def fix_netcat_colors():
        from IPython.core import prompts
        for scheme in (prompts.PColLinux, prompts.PColLightBG):
            colors = scheme.colors
            for key, val in colors.items():
                colors[key] = val.replace('\x01', '').replace('\x02', '')

    def run(self):
        if self.interactive or not sys.stdin.isatty():
            return
        key = ''
        i, _, _ = select.select([sys.stdin], [], [], 0)
        for s in i:
            if s == sys.stdin:
                key = sys.stdin.readline()[0]
                break
        if key == 'i':
            self.interact()
