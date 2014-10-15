#!/usr/bin/python3
# encoding: utf-8
# PYTHON_ARGCOMPLETE_OK
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


from email.utils import formatdate
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import veles

__doc__ += (" " * 7 +  # pylint: disable=W0622
            ("\n" + " " * 7).join(veles.__logo__.split('\n')) +
            u"\u200B\n")

try:
    import argcomplete
except:
    pass
import argparse
import atexit
import binascii
import errno
import gc
import logging
import numpy
import resource
import runpy
from six import print_

from veles.cmdline import CommandLineArgumentsRegistry
from veles.config import root
import veles.external.daemon as daemon
from veles.logger import Logger
from veles.launcher import Launcher
from veles.opencl import Device
from veles.pickle2 import setup_pickle_debug
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.thread_pool import ThreadPool

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
    """
    Start point of any VELES engine executions.
    """

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
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

    LOG_LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO,
                     "warning": logging.WARNING, "error": logging.ERROR}

    DRY_RUN_CHOICES = ["load", "init", "exec", "no"]

    SPECIAL_OPTS = ["--help", "--html-help", "--version", "--frontend"]

    @staticmethod
    def init_parser(sphinx=False):
        """
        Creates the command line argument parser.
        """

        class SortingRawDescriptionHelpFormatter(
                argparse.RawDescriptionHelpFormatter):
            def add_arguments(self, actions):
                actions = sorted(actions, key=lambda x: x.dest)
                super(SortingRawDescriptionHelpFormatter, self).add_arguments(
                    actions)

        parser = argparse.ArgumentParser(
            description=Main.LOGO if not sphinx else "",
            formatter_class=SortingRawDescriptionHelpFormatter)
        for cls in CommandLineArgumentsRegistry.classes:
            parser = cls.init_parser(parser=parser)
        parser.add_argument("--no-logo", default=False,
                            help="Do not print VELES version, copyright and "
                            "other information on startup.",
                            action='store_true')
        parser.add_argument("--version", action="store_true",
                            help="Print version number, date, commit hash and "
                            "exit.")
        parser.add_argument("--html-help", action="store_true",
                            help="Open VELES help in your web browser.")
        parser.add_argument("--frontend", action="store_true",
                            help="Open VELES command line frontend in the "
                            "default web browser and run the composed line.")
        parser.add_argument("-v", "--verbosity", type=str, default="info",
                            choices=Main.LOG_LEVEL_MAP.keys(),
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
                            help="Print the resulting workflow configuration",
                            action='store_true')
        parser.add_argument("--dry-run", default="no",
                            choices=Main.DRY_RUN_CHOICES,
                            help="no: normal work; load: stop before loading/"
                            "creating the workflow; init: stop before workflow"
                            " initialization; exec: stop before workflow "
                            "execution.")
        parser.add_argument("--visualize", default=False,
                            help="initialize, but do not run the loaded "
                            "model, show workflow graph and plots",
                            action='store_true')
        parser.add_argument("--optimize", type=str, default="no",
                            help="Do optimization of the parameters using the "
                            "genetic algorithm. Possible values: "
                            "no: off; single: local sequential optimization; "
                            "multi: use master's nodes to do the "
                            "distributed optimization (each instance will run "
                            "in standalone mode). \"single\" and \"multi\" may"
                            " end with the colon with a number; that number "
                            "sets the population size.")
        parser.add_argument("--workflow-graph", type=str, default="",
                            help="Save workflow graph to file.")
        parser.add_argument("--dump-unit-attributes", default="no",
                            help="Print unit __dict__-s after workflow "
                            "initialization, excluding large numpy arrays if "
                            "\"pretty\" is chosen.",
                            choices=['no', 'pretty', 'all'])
        parser.add_argument(
            'workflow', help='Path to Python script with the VELES model.'
            ).pretty_name = "workflow file"
        parser.add_argument(
            'config', default="-", help="Path to the configuration file"
            "(pass \"-\" to set as <workflow>_config.py)."
            ).pretty_name = "configuration file"
        arg = parser.add_argument(
            'config_list', help="Configuration overrides separated by a "
            "whitespace, for example: \nroot.global_alpha=0.006\n"
            "root.snapshot_prefix='test_pr'", nargs='*', metavar="key=value")
        arg.pretty_name = "override configuration"
        arg.important = True
        parser.add_argument("-b", "--background", default=False,
                            help="Run in background as a daemon.",
                            action='store_true')
        parser.add_argument("--disable-opencl", default=False,
                            action="store_true", help="Completely disable the "
                            "usage of OpenCL.")
        try:
            class NoEscapeCompleter(argcomplete.CompletionFinder):
                def quote_completions(self, completions, *args, **kwargs):
                    return completions
            NoEscapeCompleter()(parser)  # pylint: disable=E1102
        except:
            pass
        return parser

    @property
    def optimization(self):
        return self._optimization != "no"

    @optimization.setter
    def optimization(self, value):
        if value:
            if not self.optimization:
                raise ValueError("Genetics cannot be forced to be used")
            return
        self._optimization = "no"

    @property
    def opencl_is_enabled(self):
        return not (self.launcher.is_master or self.optimization or
                    self._disable_opencl)

    def _process_special_args(self):
        if "--frontend" in sys.argv:
            try:
                self._open_frontend()
            except KeyboardInterrupt:
                return Main.EXIT_FAILURE
            return self._process_special_args()
        if "--version" in sys.argv:
            self._print_version()
            return Main.EXIT_SUCCESS
        if "--html-help" in sys.argv:
            veles.__html__()
            return Main.EXIT_SUCCESS
        if "--help" in sys.argv:
            # help text requires UTF-8, but the default codec is ascii over ssh
            Logger.ensure_utf8_streams()
        return None

    def _open_frontend(self):
        from multiprocessing import Process, SimpleQueue
        connection = SimpleQueue()
        frontend = Process(target=self._open_frontend_process,
                           args=(connection,))
        frontend.start()
        cmdline = connection.get()
        frontend.join()
        sys.argv[1:] = cmdline.split()
        print("Running with the following command line: %s" % sys.argv)

    def _open_frontend_process(self, connection):
        if not os.path.exists(os.path.join(root.common.web.root,
                                           "frontend.html")):
            self.info("frontend.html was not found, generating it...")
            from .generate_frontend import main
            main()
            gc.collect()

        from random import randint
        port = randint(1024, 65535)
        self.info("Launching the web server on localhost:%d...", port)

        from tornado.escape import json_decode
        from tornado.ioloop import IOLoop
        import tornado.web as web

        cmdline = [""]

        class CmdlineHandler(web.RequestHandler):
            def post(self):
                try:
                    data = json_decode(self.request.body)
                    cmdline[0] = data.get("cmdline")
                    IOLoop.instance().add_callback(IOLoop.instance().stop)
                except:
                    self.exception("Frontend cmdline POST")

        app = web.Application([
            ("/cmdline", CmdlineHandler),
            (r"/(js/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/(css/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/(fonts/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/(img/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/(frontend\.html)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            ("/", web.RedirectHandler, {"url": "/frontend.html",
                                        "permanent": True}),
            ("", web.RedirectHandler, {"url": "/frontend.html",
                                       "permanent": True})
        ])
        app.listen(port)

        def open_browser():
            from veles.portable import show_file
            show_file("http://localhost:%d" % port)

        IOLoop.instance().add_callback(open_browser)
        try:
            IOLoop.instance().start()
        except KeyboardInterrupt:
            sys.stderr.write("KeyboardInterrupt\n")
        finally:
            connection.put(cmdline[0])

    def _parse_optimization(self, args):
        optparsed = args.optimize.split(':')
        if len(optparsed) > 2:
            raise ValueError("\"%s\" is not a valid optimization setting" %
                             args.optimize)
        self._optimization = optparsed[0]
        if len(optparsed) > 1:
            try:
                self._population_size = int(optparsed[1])
            except:
                raise ValueError("\"%s\" is not a valid optimization size" %
                                 optparsed[1])
        else:
            self._population_size = 0

    def _daemonize(self):
        daemon_context = daemon.DaemonContext()
        daemon_context.working_directory = os.getcwd()
        daemon_context.files_preserve = [
            int(fd) for fd in os.listdir("/proc/self/fd")
            if int(fd) > 2]
        self.info("Daemonized")
        # Daemonization happens in open()
        daemon_context.open()

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

    def _seed_random(self, rndvals):
        rndvals_split = rndvals.split(',')
        for rndval, index in zip(rndvals_split, range(len(rndvals_split))):
            try:
                binvle = binascii.unhexlify(rndval)
                rnd.get(index + 1).seed(
                    numpy.frombuffer(binvle, dtype=numpy.uint8),
                    dtype=numpy.uint8)
                continue
            except (binascii.Error, TypeError):
                pass
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
        try:
            return Snapshotter.import_(fname_snapshot)
        except FileNotFoundError:
            if fname_snapshot.strip() != "":
                self.warning("Workflow snapshot %s does not exist",
                             fname_snapshot)
            return None

    def _load(self, Workflow, **kwargs):
        self.debug("load() was called from run(), workflow class is %s",
                   str(Workflow))
        self.load_called = True
        try:
            self.launcher = Launcher()
        except:
            self.exception("Failed to create the launcher")
            sys.exit(Main.EXIT_FAILURE)
        try:
            self.workflow = self._load_workflow(self.snapshot_file_name)
            snapshot = self.workflow is not None
            if not snapshot:
                self.workflow = Workflow(self.launcher, **kwargs)
                self.info("Created %s", self.workflow)
            else:
                self.info("Loaded the workflow pickle from %s: %s",
                          self.snapshot_file_name, self.workflow)
                if self._visualization_mode:
                    self.workflow.plotters_are_enabled = True
                self.workflow.workflow = self.launcher
        except:
            self.exception("Failed to create the workflow")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)
        if ThreadPool.manhole:
            from veles.external import manhole
            manhole.WORKFLOW = self.workflow
        if self._workflow_graph:
            self.workflow.generate_graph(filename=self._workflow_graph,
                                         with_data_links=True,
                                         background='white')
        return self.workflow, snapshot

    def _main(self, **kwargs):
        if self._dry_run < 2:
            self.launcher.stop()
            return
        self.debug("main() was called from run()")
        if not self.load_called:
            self.critical("Call load() first in run()")
            sys.exit(Main.EXIT_FAILURE)
        self.main_called = True
        try:
            self.device = Device() if self.opencl_is_enabled else None
        except:
            self.exception("Failed to create the OpenCL device.")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)
        try:
            self.workflow.initialize(device=self.device, **kwargs)
        except:
            self.exception("Failed to initialize the workflow")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)
        self.debug("Workflow initialization has been completed")
        try:
            if self._dump_attrs != "no":
                self._dump_unit_attributes(self._dump_attrs == "all")
            gc.collect()
            if self._dry_run > 2:
                self.debug("Running the launcher")
                self.launcher.run()
            elif self._visualization_mode:
                self.debug("Visualizing the workflow...")
                self._visualize_workflow()
        except:
            self.exception("Failed to run the workflow")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)

    def _dump_unit_attributes(self, arrays=True):
        import veles.external.prettytable as prettytable
        from veles import Workflow
        self.debug("Dumping unit attributes of %s...", str(self.workflow))
        table = prettytable.PrettyTable("#", "unit", "attr", "value")
        table.align["#"] = "r"
        table.align["unit"] = "l"
        table.align["attr"] = "l"
        table.align["value"] = "l"
        table.max_width["value"] = 100
        for i, u in enumerate(self.workflow.units_in_dependency_order):
            for k, v in sorted(u.__dict__.items()):
                if k not in Workflow.HIDDEN_UNIT_ATTRS:
                    if not arrays and hasattr(v, "__len__") and len(v) > 32 \
                       and not isinstance(v, str) and not isinstance(v, bytes):
                        strv = "object of class %s of length %d" % (
                            repr(v.__class__.__name__), len(v))
                    else:
                        strv = repr(v)
                    table.add_row(i, u.__class__.__name__, k, strv)
        print(table)

    def _visualize_workflow(self):
        _, file_name = self.workflow.generate_graph(with_data_links=True,
                                                    background='white')
        from veles.portable import show_file
        show_file(file_name)

        import signal
        self.launcher.graphics_client.send_signal(signal.SIGUSR2)
        from twisted.internet import reactor
        reactor.callWhenRunning(self._run_workflow_plotters)
        reactor.callWhenRunning(print_, "Press Ctrl-C when you are done...")
        reactor.run()

    def _run_workflow_plotters(self):
        from veles.plotter import Plotter
        for unit in self.workflow:
            if isinstance(unit, Plotter):
                unit.run()
        # Second loop is needed to finish with PDF
        for unit in self.workflow:
            if isinstance(unit, Plotter):
                unit.last_run_time = 0
                unit.run()
                break

    def _run_core(self, wm, background):
        if self._dry_run <= 0:
            return
        if background:
            self._daemonize()
        if not self.optimization:
            from veles.genetics import fix_config
            fix_config(root)
            self.run_module(wm)
        else:
            from veles.genetics import ConfigPopulation
            ConfigPopulation(root, self, wm, self._optimization == "multi",
                             self._population_size or 50).evolve()

    def _print_logo(self, args):
        if not args.no_logo:
            try:
                print(Main.LOGO)
            except:
                print(Main.LOGO.replace("©", "(c)"))

    def _print_version(self):
        print(veles.__version__, formatdate(veles.__date__, True),
              veles.__git__)

    def print_max_rss(self):
        res = resource.getrusage(resource.RUSAGE_SELF)
        self.info("Peak resident memory used: %d Kb", res.ru_maxrss)

    def run_module(self, module):
        self.debug("Calling %s.run()...", module.__name__)
        module.run(self._load, self._main)
        if not self.main_called and self._dry_run > 2:
            self.warning("main() was not called by run() in %s",
                         module.__file__)

    """
    Basically, this is what each workflow module's run() should do.
    """
    def run_workflow(self, Workflow, kwargs_load=None, kwargs_main=None):
        # we should not set "{}" as default values because of the way
        # default values work: the dicts will be reused, not recreated
        if kwargs_load is None:
            kwargs_load = {}
        if kwargs_main is None:
            kwargs_main = {}
        self._load(Workflow, **kwargs_load)
        self._main(**kwargs_main)

    def run(self):
        """Entry point method.
        """
        ret = self._process_special_args()
        if ret is not None:
            return ret

        parser = Main.init_parser()
        args = parser.parse_args()
        fname_config = args.config
        if fname_config == "-":
            fname_config = "%s_config%s" % os.path.splitext(args.workflow)
        fname_config = os.path.abspath(fname_config)
        fname_workflow = os.path.abspath(args.workflow)
        self._visualization_mode = args.visualize
        self._workflow_graph = args.workflow_graph
        self._dry_run = Main.DRY_RUN_CHOICES.index(args.dry_run)
        self._dump_attrs = args.dump_unit_attributes
        self._disable_opencl = args.disable_opencl
        self._parse_optimization(args)

        self._print_logo(args)
        Logger.setup(level=Main.LOG_LEVEL_MAP[args.verbosity])
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

        self._run_core(wm, args.background)
        self.info("End of job")
        return Main.EXIT_SUCCESS


def __run__():
    retcode = Main().run()
    sys.exit(retcode)

if __name__ == "__main__":
    __run__()
