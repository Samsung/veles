"""
Created on Apr 7, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
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
        self.shell_ = InteractiveShellEmbed(config=self.cfg,
                                            banner1=Shell.BANNER1,
                                            banner2=Shell.BANNER2)

    def run(self):
        key = ''
        i, _, _ = select.select([sys.stdin], [], [], 0)
        for s in i:
            if s == sys.stdin:
                key = sys.stdin.readline()[0]
                break
        if key == 'i':
            workflow = self.workflow  # pylint: disable=W0612
            units = self.workflow.units  # pylint: disable=W0612
            self.shell_()
