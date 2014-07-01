#!/usr/bin/python3
# encoding: utf-8
'''
This script starts the Veles platform and executes the user's model (workflow).

Contact:
    g.kuznetsov@samsung.com
    v.markovtsev@samsung.com

.. argparse::
   :module: scripts.velescli
   :func: create_args_parser_sphinx
   :prog: velescli

   ::


'''
__module_veles_logo__ = \
    r"        _   _ _____ _     _____ _____\n" + \
    r"       | | | |  ___| |   |  ___/  ___|  " + \
    r"Version 0.3.0 Wed, 23 Apr 2014 14:46:21 +0400\n" + \
    r"       | | | | |__ | |   | |__ \ `––.   " + \
    r"Copyright © 2013 Samsung Electronics Co., Ltd.\n" + \
    r"       | | | |  __|| |   |  __| `––. \  " + \
    r"All rights reserved. Any unauthorized use of\n" + \
    r"       \ \_/ / |___| |___| |___/\__/ /  " + \
    r"this software is strictly prohibited and is\n" + \
    r"        \___/\____/\_____|____/\____/   " + \
    r"a subject of your country's laws.\n" + \
    r"       \u200B\n"

__doc__ += __module_veles_logo__  # nopep8

try:
    import argcomplete
except:
    pass
import argparse
import atexit
import bz2
from email.utils import formatdate
import errno
import gzip
import logging
import lzma
import numpy
import os
import resource
import runpy
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import veles
from veles.config import root
from veles.graphics_server import GraphicsServer
from veles.logger import Logger
from veles.launcher import Launcher
from veles.opencl import Device
from veles.opencl_units import OpenCLUnit
from veles.pickle2 import pickle, setup_pickle_debug
import veles.random_generator as rnd

if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622
    IsADirectoryError = IOError  # pylint: disable=W0622
    PermissionError = IOError  # pylint: disable=W0622


def create_args_parser_sphinx():
    """
    This is a top-level function to please Sphinx.
    """
    return Main.init_parser(True)


class Main(Logger):
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    LOGO_PLAIN = r" _   _ _____ _     _____ _____  " "\n" \
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

    LOGO = LOGO_COLORED if sys.stdout.isatty() else LOGO_PLAIN

    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}

    CODECS = {
        ".pickle": lambda name: open(name, "rb"),
        ".gz": lambda name: gzip.GzipFile(name, "rb"),
        ".bz2": lambda name: bz2.BZ2File(name, "rb"),
        ".xz": lambda name: lzma.LZMAFile(name, "rb")
    }

    @staticmethod
    def init_parser(sphinx=False):
        """
        Creates the command line argument parser.
        """
        parser = argparse.ArgumentParser(
            description=Main.LOGO if not sphinx else "",
            formatter_class=argparse.RawDescriptionHelpFormatter)
        parser = Launcher.init_parser(parser=parser)
        parser = Device.init_parser(parser=parser)
        parser = OpenCLUnit.init_parser(parser=parser)
        parser = GraphicsServer.init_parser(parser)
        parser.add_argument("--no-logo", default=False,
                            help="Do not print VELES version, copyright and "
                            "other information on startup.",
                            action='store_true')
        parser.add_argument("-v", "--verbose", type=str, default="info",
                            choices=Main.LOG_LEVEL_MAP.keys(),
                            help="set logging verbosity level")
        parser.add_argument("--debug", type=str, default="",
                            help="set DEBUG logging level for these classes "
                                 "(separated by commas)")
        parser.add_argument("--debug-pickle", default=False,
                            help="turn on pickle diagnostics",
                            action='store_true')
        parser.add_argument("-r", "--random-seed", type=str,
                            default="/dev/urandom:16",
                            help="set random seed, e.g. "
                                 "veles/samples/seed:1024,:1024 or "
                                 "/dev/urandom:16:uint32")
        parser.add_argument('-w', '--snapshot', default="",
                            help='workflow snapshot')
        parser.add_argument("--dump-config", default=False,
                            help="print the resulting workflow configuration",
                            action='store_true')
        parser.add_argument("--dry-run", default=False,
                            help="do not run the loaded model",
                            action='store_true')
        parser.add_argument('workflow',
                            help='path to the Python script with workflow')
        parser.add_argument('config', default="-",
                            help='path to the configuration file\
                            (write "-" to search in as <WORKFLOW>_config.py)')
        parser.add_argument('config_list',
                            help="list of configuration overloads like: \n"
                            "root.global_alpha=0.006\n"
                            "root.snapshot_prefix='test_pr'",
                            nargs='*', metavar="configs...")
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions
            NoEscapeCompleter()(parser)  # pylint: disable=E1102
        except:
            pass
        return parser

    def _load_model(self, fname_workflow, fname_snapshot):
        self.debug("Loading the model \"%s\"...", fname_workflow)
        self.snapshot_file_name = fname_snapshot
        self.load_called = False
        self.main_called = False
        try:
            sys.path.insert(0, os.path.dirname(fname_workflow))
            module = __import__(
                os.path.splitext(os.path.basename(fname_workflow))[0])
            del sys.path[0]
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
            self.exception("Failed to load the workflow \"%s\"",
                           fname_workflow)
            sys.exit(Main.EXIT_FAILURE)
        return module

    def _apply_config(self, fname_config, config_list):
        self.debug("Applying the configuration from %s...",
                   fname_config)
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
        override_cfg = "\n".join(config_list)
        self.debug("Overriding the configuration with %s", override_cfg)
        try:
            exec(override_cfg)
        except:
            self.exception("Invalid configuration overloads")
            sys.exit(Main.EXIT_FAILURE)

    def _run_workflow(self, module):
        self.debug("Calling %s.run()...", module.__name__)
        module.run(self._load, self._main)
        if not self.main_called:
            self.warning("main() was not called by run() in %s",
                         module.__file__)

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
            if fname == "-":
                try:
                    rnd.get(index + 1).seed(None)
                except:
                    self.exception("Failed to seed the random generator %d "
                                   "with the last used seed.", index + 1)
                    sys.exit(Main.EXIT_FAILURE)
                continue
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

            self.debug("Seeding with %d samples of type %s from %s to %d",
                       count, str(dtype), fname, index + 1)
            try:
                rnd.get(index + 1).seed(numpy.fromfile(fname, dtype=dtype,
                                                       count=count),
                                        dtype=dtype)
            except:
                self.exception("Failed to seed the random generator with %s",
                               fname)
                sys.exit(Main.EXIT_FAILURE)

    def _load_workflow(self, fname_snapshot):
        fname_snapshot = fname_snapshot.strip()
        if os.path.exists(fname_snapshot):
            _, ext = os.path.splitext(fname_snapshot)
            codec = Main.CODECS[ext]
            with codec(fname_snapshot) as fin:
                return pickle.load(fin)
        if fname_snapshot != "":
            self.warning("Workflow snapshot %s does not exist",
                         fname_snapshot)
        return None

    def _load(self, Workflow, **kwargs):
        self.debug("load() was called from run(), workflow class is %s",
                   str(Workflow))
        self.load_called = True
        try:
            self.launcher = Launcher()
            self.workflow = self._load_workflow(self.snapshot_file_name)
            snapshot = self.workflow is not None
            if not snapshot:
                self.workflow = Workflow(self.launcher, **kwargs)
            else:
                self.info("Loaded the workflow pickle from %s",
                          self.snapshot_file_name)
                self.workflow.workflow = self.launcher
        except:
            self.exception("Failed to create the workflow")
            sys.exit(Main.EXIT_FAILURE)
        return self.workflow, snapshot

    def _main(self, **kwargs):
        self.debug("main() was called from run()")
        if not self.load_called:
            self.critical("Call load() first in run()")
            raise RuntimeError()
        self.main_called = True
        self.device = None if self.launcher.is_master else Device()
        try:
            self.workflow.initialize(device=self.device, **kwargs)
            self.debug("Workflow initialization has been completed."
                       "Running the launcher.")
            self.launcher.run()
        except:
            self.exception("Failed to run the workflow")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)

    def _print_logo(self, args):
        if not args.no_logo:
            try:
                print(Main.LOGO)
            except:
                print(Main.LOGO.replace("©", "(c)"))

    def print_max_rss(self):
        res = resource.getrusage(resource.RUSAGE_SELF)
        self.info("Peak resident memory used: %d Kb", res.ru_maxrss)

    def run(self):
        """VELES Machine Learning Platform Command Line Interface
        """
        parser = Main.init_parser()
        args = parser.parse_args()
        fname_config = args.config
        if fname_config == "-":
            fname_config = "%s_config%s" % os.path.splitext(args.workflow)
        fname_config = os.path.abspath(fname_config)
        fname_workflow = os.path.abspath(args.workflow)

        self._print_logo(args)
        Logger.setup(level=Main.LOG_LEVEL_MAP[args.verbose])
        for name in filter(str.strip, args.debug.split(',')):
            logging.getLogger(name).setLevel(logging.DEBUG)
        self._seed_random(args.random_seed)
        if args.debug_pickle:
            setup_pickle_debug()
        atexit.register(self.print_max_rss)

        wm = self._load_model(fname_workflow, args.snapshot)
        self._apply_config(fname_config, args.config_list)
        if args.dump_config:
            root.print_config()
        if not args.dry_run:
            self._run_workflow(wm)
            self.info("End of job")
        return Main.EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(Main().run())
