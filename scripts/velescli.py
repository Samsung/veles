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
import errno
import logging
import os
import runpy
import sys


import veles
import veles.logger
from veles.launcher import Launcher
from veles.opencl import Device


class Main(veles.logger.Logger):
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    LOGO = r" _   _ _____ _     _____ _____  " "\n" \
            r"| | | |  ___| |   |  ___/  ___| " + \
            (" Version %s\n" % veles.__version__) + \
            r"| | | | |__ | |   | |__ \ `--.  " + \
            (" Copyright %s\n" % veles.__copyright__) + \
            r"| | | |  __|| |   |  __| `--. \ " \
            " All rights reserved. Any unauthorized use of\n" \
            r"\ \_/ / |___| |___| |___/\__/ / " \
            " this software is strictly prohibited and\n" \
            r" \___/\____/\_____|____/\____/  " \
            " is a subject of your country's laws.\n"

    def run(self):
        """VELES Machine Learning Platform Command Line Interface
        """
        parser = argparse.ArgumentParser(
            description=Main.LOGO,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        parser = Launcher.init_parser(parser=parser)
        parser = Device.init_parser(parser=parser)
        parser.add_argument("--no-logo", default=False,
                            help="Do not print VELES version, copyright and "
                            "other information on startup.",
                            action='store_true')
        log_level_map = {"debug": logging.DEBUG, "info": logging.INFO,
                         "warning": logging.WARNING, "error": logging.ERROR}
        parser.add_argument("-v", "--verbose", type=str, default="info",
                            choices=log_level_map.keys(),
                            help="set verbosity level [default: %(default)s]")
        parser.add_argument('workflow',
                            help='path to the Python script with workflow')
        parser.add_argument('config',
                            help='path to the configuration file')
        parser.add_argument('config_list',
                            help="list of configuration overloads like: \n"
                            "root.global_alpha=0.006\n"
                            "root.snapshot_prefix='test_pr'",
                            nargs='*', metavar="config,")

        args = parser.parse_args()
        fname_workflow = os.path.abspath(args.workflow)
        fname_config = os.path.abspath(args.config)
        config_list = args.config_list

        logging.basicConfig(level=log_level_map[args.verbose])

        if not args.no_logo:
            print(Main.LOGO)
        try:
            runpy.run_path(fname_config)
        except FileNotFoundError:
            self.exception("Configuration does not exist: \"%s\"",
                           fname_config)
            return errno.ENOENT
        except IsADirectoryError:
            self.exception("Configuration \"%s\" is a directory", fname_config)
            return errno.EISDIR
        except PermissionError:
            self.exception("Cannot read configuration \"%s\"", fname_config)
            return errno.EACCES
        except:
            self.exception("Failed to apply the configuration \"%s\"",
                           fname_config)
            return Main.EXIT_FAILURE

        try:
            exec("\n".join(config_list))
        except:
            self.exception("Invalid configuration overloads")
            return Main.EXIT_FAILURE

        try:
            runpy.run_path(fname_workflow, run_name="__main__")
        except FileNotFoundError:
            self.exception("Workflow does not exist: \"%s\"", fname_workflow)
            return errno.ENOENT
        except IsADirectoryError:
            self.exception("Workflow \"%s\" is a directory", fname_workflow)
            return errno.EISDIR
        except PermissionError:
            self.exception("Cannot read workflow \"%s\"", fname_workflow)
            return errno.EACCES
        except:
            self.exception("Failed to run the workflow \"%s\"",
                           fname_workflow)
            return Main.EXIT_FAILURE

        return Main.EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(Main().run())
