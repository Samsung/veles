#!/usr/bin/python3
# encoding: utf-8
'''
veles is python script which starts platform and
executes user script (called experiment)

@author:     Gennady Kuznetsov
@copyright:  Copyright 2013 Samsung R&D Institute Russia
@contact:    g.kuznetsov@samsung.com
'''


import argparse
from email.utils import formatdate
import errno
import logging
import numpy
import os
import runpy
import sys

from veles.config import root
import veles.logger
from veles.launcher import Launcher
from veles.opencl import Device
import veles.rnd as rnd


class Main(veles.logger.Logger):
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    LOGO = r" _   _ _____ _     _____ _____  " "\n" \
           r"| | | |  ___| |   |  ___/  ___| " + \
           (" Version %s" % veles.__version__) + \
           (" %s\n" % formatdate(veles.__date__, True)) + \
           r"| | | | |__ | |   | |__ \ `--.  " + \
           (" Copyright %s\n" % veles.__copyright__) + \
           r"| | | |  __|| |   |  __| `--. \ " \
           " All rights reserved. Any unauthorized use of\n" \
           r"\ \_/ / |___| |___| |___/\__/ / " \
           " this software is strictly prohibited and is\n" \
           r" \___/\____/\_____|____/\____/  " \
           " a subject of your country's laws.\n"

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

    LOGO_OPT = LOGO_COLORED if sys.stdout.isatty() else LOGO

    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}

    def _init_parser(self):
        parser = argparse.ArgumentParser(
            description=Main.LOGO_OPT,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        parser = Launcher.init_parser(parser=parser)
        parser = Device.init_parser(parser=parser)
        parser.add_argument("--no-logo", default=False,
                            help="Do not print VELES version, copyright and "
                            "other information on startup.",
                            action='store_true')
        parser.add_argument("-v", "--verbose", type=str, default="info",
                            choices=Main.LOG_LEVEL_MAP.keys(),
                            help="set verbosity level [default: %(default)s]")
        parser.add_argument("-r", "--random-seed", type=str,
                            default="/dev/random:16",
                            help="set random seed, e.g. "
                                 "veles/samples/seed:1024,:1024 or "
                                 "/dev/urandom:16:uint32")
        parser.add_argument('workflow',
                            help='path to the Python script with workflow')
        parser.add_argument('config', default="-",
                            help='path to the configuration file')
        parser.add_argument('config_list',
                            help="list of configuration overloads like: \n"
                            "root.global_alpha=0.006\n"
                            "root.snapshot_prefix='test_pr'",
                            nargs='*', metavar="configs...")
        return parser

    def _apply_config(self, fname_config, config_list):
        try:
            runpy.run_path(fname_config)
        except FileNotFoundError:
            self.exception("Configuration does not exist: \"%s\"",
                           fname_config)
            sys.exit(errno.ENOENT)
        except IsADirectoryError:
            self.exception("Configuration \"%s\" is a directory", fname_config)
            sys.exit(errno.EISDIR)
        except PermissionError:
            self.exception("Cannot read configuration \"%s\"", fname_config)
            sys.exit(errno.EACCES)
        except:
            self.exception("Failed to apply the configuration \"%s\"",
                           fname_config)
            sys.exit(Main.EXIT_FAILURE)
        try:
            exec("\n".join(config_list))
        except:
            self.exception("Invalid configuration overloads")
            sys.exit(Main.EXIT_FAILURE)

    def _seed_random(self, rndvals):
        rndvals_split = rndvals.split(',')
        for rndval, index in zip(rndvals_split, range(len(rndvals_split))):
            vals = rndval.split(':')
            fname = vals[0]
            if fname == "":
                if index > 1:
                    fname = rndvals_split[0].split(':')[0] + str(index)
                else:
                    self.critical("Random generator file name is empty")
                    sys.exit(errno.ENOENT)
            if not os.path.isabs(fname):
                new_fname = os.path.abspath(fname)
                if os.path.exists(new_fname):
                    fname = new_fname
                else:
                    fname = os.path.join(root.common.veles_dir, fname)
                    if not os.path.exists(fname):
                        self.critical("Neither %s nor %s exist. Cannot seed "
                                      "the random generator.", new_fname,
                                      fname)
                        sys.exit(errno.ENOENT)
            count = int(vals[1]) if len(vals) > 1 else 16
            dtype = numpy.dtype(vals[2]) if len(vals) > 2 else numpy.int32

            rnd_name = "default"
            if index > 1:
                rnd_name += str(index)
            try:
                rnd.__dict__[rnd_name].seed(numpy.fromfile(fname, dtype=dtype,
                                                           count=count),
                                            dtype=dtype)
            except:
                self.exception("Failed to seed the random generator with %s",
                               fname)
                sys.exit(Main.EXIT_FAILURE)

    def _run_workflow(self, fname_workflow):
        try:
            runpy.run_path(fname_workflow, run_name="__main__")
        except FileNotFoundError:
            self.exception("Workflow does not exist: \"%s\"", fname_workflow)
            sys.exit(errno.ENOENT)
        except IsADirectoryError:
            self.exception("Workflow \"%s\" is a directory", fname_workflow)
            sys.exit(errno.EISDIR)
        except PermissionError:
            self.exception("Cannot read workflow \"%s\"", fname_workflow)
            sys.exit(errno.EACCES)
        except:
            self.exception("Failed to run the workflow \"%s\"",
                           fname_workflow)
            sys.exit(Main.EXIT_FAILURE)

    def run(self):
        """VELES Machine Learning Platform Command Line Interface
        """
        parser = self._init_parser()
        args = parser.parse_args()
        fname_config = args.config
        if fname_config == "-":
            fname_config = "%s_config%s" % os.path.splitext(args.workflow)

        if not args.no_logo:
            print(Main.LOGO_OPT)
        logging.basicConfig(level=Main.LOG_LEVEL_MAP[args.verbose])
        self._seed_random(args.random_seed)

        self._apply_config(os.path.abspath(fname_config), args.config_list)
        self._run_workflow(os.path.abspath(args.workflow))

        self.info("End of job")
        return Main.EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(Main().run())
