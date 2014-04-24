#!/usr/bin/python3
# encoding: utf-8
'''
This scripts starts the Veles platform and executes the user's workflow
(called model).

@copyright:  Copyright 2013 Samsung Electronics Co., Ltd.
@contact:    g.kuznetsov@samsung.com
'''


try:
    import argcomplete
except:
    pass
import argparse
from email.utils import formatdate
import errno
import logging
import numpy
import os
from six.moves import cPickle as pickle
import runpy
import six
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import veles
from veles.config import root
from veles.logger import Logger
from veles.launcher import Launcher
from veles.opencl import Device
import veles.rnd as rnd

if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622
    IsADirectoryError = IOError  # pylint: disable=W0622
    PermissionError = IOError  # pylint: disable=W0622


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

    def _init_parser(self):
        parser = argparse.ArgumentParser(
            description=Main.LOGO,
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
        parser.add_argument("--debug", type=str, default="",
                            help="set DEBUG logging level for these names "
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
        parser.add_argument('workflow',
                            help='path to the Python script with workflow')
        parser.add_argument('config', default="-",
                            help='path to the configuration file')
        parser.add_argument('config_list',
                            help="list of configuration overloads like: \n"
                            "root.global_alpha=0.006\n"
                            "root.snapshot_prefix='test_pr'",
                            nargs='*', metavar="configs...")
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions
            NoEscapeCompleter()(parser)
        except:
            pass
        return parser

    def _load_model(self, fname_workflow, fname_snapshot):
        self.debug("Loading the model \"%s\"...", fname_workflow)
        self.snapshot_file_name = fname_snapshot
        self.load_called = False
        self.main_called = False
        try:
            sys.path = [os.path.dirname(fname_workflow)] + sys.path
            module = __import__(
                os.path.splitext(os.path.basename(fname_workflow))[0])
            sys.path = sys.path[1:]
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
            self.debug("Seeding with %d samples of type %s from %s",
                       count, str(dtype), fname)
            try:
                rnd.__dict__[rnd_name].seed(numpy.fromfile(fname, dtype=dtype,
                                                           count=count),
                                            dtype=dtype)
            except:
                self.exception("Failed to seed the random generator with %s",
                               fname)
                sys.exit(Main.EXIT_FAILURE)

    def _load_workflow(self, fname_snapshot):
        fname_snapshot = fname_snapshot.strip()
        if os.path.exists(fname_snapshot):
            with open(fname_snapshot, "rb") as fin:
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
            self.device = None if self.launcher.is_master else Device()
            self.workflow = self._load_workflow(self.snapshot_file_name)
            snapshot = self.workflow is not None
            if not snapshot:
                self.workflow = Workflow(self.launcher, device=self.device,
                                         **kwargs)
            else:
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

    def _set_pickle_debug(self, args):
        if not args.debug_pickle:
            return
        if not six.PY3:
            self.warning("Pickle debugging is only available for Python 3.x")
            return

        def dump(obj, file, protocol=None, fix_imports=True):
            pickle._Pickler(file, protocol, fix_imports=fix_imports).dump(obj)

        pickle.dump = dump
        orig_save = pickle._Pickler.save

        def save(self, obj):
            try:
                orig_save(self, obj)
            except:
                import traceback
                import pdb
                print("\033[1;31mPickle failure\033[0m")
                traceback.print_exc()
                pdb.set_trace()

        pickle._Pickler.save = save

    def run(self):
        """VELES Machine Learning Platform Command Line Interface
        """
        parser = self._init_parser()
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
        self._set_pickle_debug(args)

        wm = self._load_model(fname_workflow, args.snapshot)
        self._apply_config(fname_config, args.config_list)
        self._run_workflow(wm)

        self.info("End of job")
        return Main.EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(Main().run())
