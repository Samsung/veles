"""
Created on Jul 2, 2014

Base class for __main__'s Main class and others which are topmost script
classes.

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


try:
    import argcomplete
except:
    pass
from argparse import RawDescriptionHelpFormatter, ArgumentParser, ArgumentError
from email.utils import formatdate
import logging
import sys

import veles
from veles.compat import from_none


class CommandLineArgumentsRegistry(type):
    """
    Metaclass to accumulate command line options from scattered classes for
    velescli's upmost argparse.
    """
    classes = []

    def __init__(cls, name, bases, clsdict):
        super(CommandLineArgumentsRegistry, cls).__init__(name, bases, clsdict)
        # if the class does not have it's own init_parser(), no-op
        init_parser = clsdict.get('init_parser', None)
        if init_parser is None:
            return
        # early check for the method existence
        if not isinstance(init_parser, staticmethod):
            raise TypeError("init_parser must be a static method since the "
                            "class has CommandLineArgumentsRegistry metaclass")
        CommandLineArgumentsRegistry.classes.append(cls)


class CommandLineBase(object):
    """
    Start point of any VELES engine executions.
    """

    LOGO_PLAIN = veles.__logo__

    LOGO_COLORED = "\033" r"[1;32m _   _ _____ _     _____ _____  " \
                   "\033[0m\n" \
                   "\033" r"[1;32m| | | |  ___| |   |  ___/  ___| " \
                   "\033[0m" + \
                   (" Version \033[1;36m%s\033[0m" % veles.__version__) + \
                   (" %s\n" % formatdate(veles.__date__, True)) + \
                   "\033" r"[1;32m| | | | |__ | |   | |__ \ `--.  " \
                   "\033[0m" + ("\033[0;37m Copyright %s\033[0m\n" %
                                veles.__copyright__) + \
                   "\033" r"[1;32m| | | |  __|| |   |  __| `--. \ " "\033[0m" \
                   "\033[0;37m All rights reserved. Any unauthorized use of" \
                   "\033[0m\n" \
                   "\033" r"[1;32m\ \_/ / |___| |___| |___/\__/ / " "\033[0m" \
                   "\033[0;37m this software is strictly prohibited and is" \
                   "\033[0m\n" \
                   "\033" r"[1;32m \___/\____/\_____|____/\____/  " "\033[0m" \
                   "\033[0;37m a subject of your country's laws.\033[0m\n"

    LOGO = LOGO_COLORED if sys.stdout.isatty() else LOGO_PLAIN
    DRY_RUN_CHOICES = ["load", "init", "exec", "no"]
    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}
    SPECIAL_OPTS = ["--help", "--html-help", "--version", "--frontend",
                    "--dump-config"]

    class SortingRawDescriptionHelpFormatter(RawDescriptionHelpFormatter):
        def add_arguments(self, actions):
            actions = sorted(actions, key=lambda x: x.dest)
            super(CommandLineBase.SortingRawDescriptionHelpFormatter,
                  self).add_arguments(actions)

    @staticmethod
    def init_parser(sphinx=False, ignore_conflicts=False):
        """
        Creates the command line argument parser.
        """

        parser = ArgumentParser(
            description=CommandLineBase.LOGO if not sphinx else "",
            formatter_class=CommandLineBase.SortingRawDescriptionHelpFormatter)
        for cls in CommandLineArgumentsRegistry.classes:
            try:
                parser = cls.init_parser(parser=parser)
            except ArgumentError as e:
                if not ignore_conflicts:
                    raise from_none(e)
        parser.add_argument("--no-logo", default=False,
                            help="Do not print VELES version, copyright and "
                                 "other information on startup.",
                            action='store_true')
        parser.add_argument("--version", action="store_true",
                            help="Print version number, date, commit hash and "
                                 "exit.")
        parser.add_argument("--html-help", action="store_true",
                            help="Open VELES help in your web browser.")
        parser.add_argument(
            "--frontend", action="store_true",
            help="Open VELES command line frontend in the default web browser "
                 "and run the composed line.")
        parser.add_argument("-v", "--verbosity", type=str, default="info",
                            choices=CommandLineBase.LOG_LEVEL_MAP.keys(),
                            help="Set the logging verbosity level.")
        parser.add_argument("--debug", type=str, default="",
                            help="Set DEBUG logging level for these classes "
                                 "(separated by comma)")
        parser.add_argument("--debug-pickle", default=False,
                            help="Turn on pickle diagnostics.",
                            action='store_true')
        parser.add_argument("-r", "--random-seed", type=str,
                            default="/dev/urandom:16",
                            help="Set the random generator seed, e.g. "
                                 "veles/samples/seed:1024,:1024 or "
                                 "/dev/urandom:16:uint32 or "
                                 "hex string with even number of digits")
        parser.add_argument('-w', '--snapshot', default="",
                            help='workflow snapshot')
        parser.add_argument("--dump-config", default=False,
                            help="Print the initial global configuration",
                            action='store_true')
        parser.add_argument("--dry-run", default="no",
                            choices=CommandLineBase.DRY_RUN_CHOICES,
                            help="no: normal work; load: stop before loading/"
                                 "creating the workflow; init: stop before "
                                 "workflow initialization; exec: stop before "
                                 "workflow execution.")
        parser.add_argument("--visualize", default=False,
                            help="initialize, but do not run the loaded "
                                 "model, show workflow graph and plots",
                            action='store_true')
        parser.add_argument(
            "--optimize", type=str, default="no",
            help="Do optimization of the parameters using the genetic "
                 "algorithm. Possible values: no: off; single: "
                 "local sequential optimization; multi: use master's nodes to "
                 "do the distributed optimization (each instance will run in "
                 "standalone mode). \"single\" and \"multi\" may end with the "
                 "colon with a number; that number sets the population size.")
        parser.add_argument("--workflow-graph", type=str, default="",
                            help="Save workflow graph to file.")
        parser.add_argument("--dump-unit-attributes", default="no",
                            help="Print unit __dict__-s after workflow "
                                 "initialization, excluding large numpy arrays"
                                 " if \"pretty\" is chosen.",
                            choices=['no', 'pretty', 'all'])
        parser.add_argument(
            'workflow', help='Path to Python script with the VELES model.'
        ).pretty_name = "workflow file"
        parser.add_argument(
            'config', help="Path to the configuration file"
                           "(pass \"-\" to set as <workflow>_config.py)."
        ).pretty_name = "configuration file"
        arg = parser.add_argument(
            'config_list',
            help="Configuration overrides separated by a whitespace, for "
                 "example: \nroot.global_alpha=0.006\n "
                 "root.snapshot_prefix='test_pr'", nargs='*',
            metavar="key=value")
        arg.pretty_name = "override configuration"
        arg.important = True
        parser.add_argument("-b", "--background", default=False,
                            help="Run in background as a daemon.",
                            action='store_true')
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions

            NoEscapeCompleter()(parser)  # pylint: disable=E1102
        except:
            pass
        return parser
