#!/usr/bin/python3
# encoding: utf-8
# PYTHON_ARGCOMPLETE_OK
u"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 25, 2013

This is the only entry point of any VELES-based execution.

Contact:
    * g.kuznetsov@samsung.com
    * v.markovtsev@samsung.com


.. argparse::
   :module: veles.__main__
   :func: create_args_parser_sphinx
   :prog: veles

   ::

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


import sys
__unittest = "unittest" in sys.modules
import atexit
import binascii
from collections import namedtuple
from email.utils import formatdate
import errno
import gc
import logging
import numpy
import os
import resource
import runpy
from six import print_, StringIO, PY3, string_types

if PY3:
    from urllib.parse import splittype
else:
    from urllib import splittype
import wget

import veles


def unload_unittest():
    if not __unittest and "unittest" in sys.modules:
        # Ensure unittest package is unloaded if it should be
        for k in [k for k in sys.modules if k.startswith("unittest")]:
            del sys.modules[k]
unload_unittest()

from veles.config import root
from veles.cmdline import CommandLineBase
from veles.compat import from_none, FileNotFoundError, IsADirectoryError, \
    PermissionError
from veles.external import daemon
from veles.import_file import get_file_package_and_module, \
    import_file_as_package, import_file_as_module
from veles.logger import Logger
from veles.launcher import Launcher
from veles.memory import Watcher
from veles.pickle2 import setup_pickle_debug
from veles import prng
from veles.snapshotter import SnapshotterToDB, SnapshotterToFile
from veles.thread_pool import ThreadPool
import veles.accelerated_units  # do not remove or options like --force-numpy
# or --sync-run in accelerated_units will disappear
import veles.loader.base  # do not remove or options like --train-ratio
# will disappear

unload_unittest()

__doc__ += (" " * 7 +  # pylint: disable=W0622
            ("\n" + " " * 7).join(veles.__logo__.split('\n')) +
            u"\u200B\n")


def create_args_parser_sphinx():
    """
    This is a top-level function to please Sphinx.
    """
    return CommandLineBase.init_parser(True)


OptimizationTuple = namedtuple("OptimizationTuple", ("multi", "size"))
EnsembleTrainTuple = namedtuple("EnsembleTrainTuple", ("size", "train_ratio"))
EnsembleTestTuple = namedtuple("EnsembleTestTuple", ("input_file",))


class Main(Logger, CommandLineBase):
    """
    Entry point of any VELES engine executions.
    """

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    registered_print_max_rss = False

    def __init__(self, interactive=False, *args, **kwargs):
        Main.setup_argv(not interactive, True, *args, **kwargs)
        super(Main, self).__init__()
        self._interactive = interactive
        self._ensemble_train = None
        self._ensemble_test = None
        self._optimization = None

    @property
    def interactive(self):
        return self._interactive

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if value is None:
            self._optimization = None
            return
        if not isinstance(value, tuple) or len(value) != 2 or \
                not isinstance(value[0], bool) or \
                not isinstance(value[1], int):
            raise ValueError("Invalid optimization value")
        if value[1] < 5:
            raise ValueError("Population size may not be less than 5")
        self._optimization = OptimizationTuple(*value)

    @property
    def ensemble_train(self):
        return self._ensemble_train

    @ensemble_train.setter
    def ensemble_train(self, value):
        if value is None:
            self._ensemble_train = None
            return
        if not isinstance(value, tuple) or len(value) != 2 or \
                not isinstance(value[0], int) or \
                not isinstance(value[1], (float, int)):
            raise ValueError("Invalid ensemble_train value")
        ratio = value[1]
        if ratio <= 0 or ratio > 1:
            raise ValueError(
                "Training set ratio must be in (0, 1] (got %s)" % ratio)
        self._ensemble_train = EnsembleTrainTuple(*value)

    @property
    def ensemble_test(self):
        return self._ensemble_test

    @ensemble_test.setter
    def ensemble_test(self, value):
        if value is None:
            self._ensemble_test = None
            return
        if not isinstance(value, string_types):
            raise TypeError(
                "ensemble_test must be a string (got %s)" % type(value))
        self._ensemble_test = EnsembleTestTuple(value)

    @property
    def regular(self):
        return not self.optimization and not self.ensemble_train and \
            not self.ensemble_test

    def _process_special_args(self):
        if "--frontend" in sys.argv:
            try:
                self._open_frontend()
            except KeyboardInterrupt:
                return Main.EXIT_FAILURE
            return self._process_special_args()
        if self.interactive:
            for opt in "forge", "--version", "--help", "--dump-config":
                if opt in self.argv:
                    raise ValueError(
                        "\"%s\" is not supported in interactive mode" % opt)
            return None
        if len(sys.argv) > 1 and sys.argv[1] == "forge":
            from veles.forge.forge_client import __run__ as forge_run
            del sys.argv[1]
            action = sys.argv[1]
            try:
                forge_run()
                return Main.EXIT_SUCCESS
            except Exception as e:
                if isinstance(e, SystemExit):
                    raise from_none(e)
                self.exception("Failed to run forge %s", action)
                return Main.EXIT_FAILURE
        if "--version" in sys.argv:
            self._print_version()
            return Main.EXIT_SUCCESS
        if "--html-help" in sys.argv:
            veles.__html__()
            return Main.EXIT_SUCCESS
        if "--help" in sys.argv:
            # help text requires UTF-8, but the default codec is ascii over ssh
            Logger.ensure_utf8_streams()
        if "--dump-config" in sys.argv:
            self.info("Scanning for the plugins...")
            self.debug("Loaded plugins: %s", veles.__plugins__)
            root.print_()
            return Main.EXIT_SUCCESS
        return None

    def _open_frontend(self):
        from multiprocessing import Process, SimpleQueue

        connection = SimpleQueue()
        frontend = Process(
            target=self._open_frontend_process,
            args=(connection, [k for k in sys.argv[1:] if k != "--frontend"]))
        frontend.start()
        cmdline = connection.get()
        frontend.join()
        if self.interactive:
            argv_backup = list(sys.argv)
        sys.argv[1:] = cmdline.split()
        Main.setup_argv(True, True)
        if self.interactive:
            sys.argv = argv_backup
        print("Running with the following command line: %s" % sys.argv)

    def _open_frontend_process(self, connection, argv):
        if not os.path.exists(os.path.join(root.common.web.root,
                                           "frontend.html")):
            self.info("frontend.html was not found, generating it...")
            from veles.scripts.generate_frontend import main

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

        class FrontendHandler(web.RequestHandler):
            def get(self):
                self.render("frontend.html", cmdline=" ".join(argv))

        app = web.Application([
            ("/cmdline", CmdlineHandler),
            (r"/((js|css|fonts|img|maps)/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/frontend\.html", FrontendHandler),
            ("/", web.RedirectHandler, {"url": "/frontend.html",
                                        "permanent": True}),
            ("", web.RedirectHandler, {"url": "/frontend.html",
                                       "permanent": True})
        ], template_path=root.common.web.root)
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
        if not optparsed[0]:
            return
        if optparsed[0] not in ("single", "multi"):
            raise ValueError(
                "Ivalid optimization type \"%s\", may be either \"single\" or "
                "\"multi\"" % optparsed[0])
        if len(optparsed) != 2:
            raise ValueError("\"%s\" is not a valid optimization setting" %
                             args.optimize)
        try:
            self.optimization = optparsed[0] == "multi", int(optparsed[1])
        except ValueError:
            raise from_none(ValueError(
                "\"%s\" is not a valid optimization size" % optparsed[1]))

    def _parse_ensemble_train(self, args):
        if args.ensemble_train is None:
            return

        optparsed = args.ensemble_train.split(":")
        if len(optparsed) != 2:
            raise ValueError(
                "--ensemble-train must be specified as"
                "<number of instances>:<training set ratio>")
        try:
            self.ensemble_train = int(optparsed[0]), float(optparsed[1])
        except ValueError:
            raise from_none(
                "Failed to parse ensemble parameters from (%s, %s)" %
                optparsed)

    def _parse_ensemble_test(self, args):
        if args.ensemble_test is None:
            return
        if self.ensemble_train is not None:
            raise ValueError(
                "--ensemble-train and --ensemble-test may not be used "
                "together")
        self.ensemble_test = args.ensemble_test

    def _daemonize(self):
        daemon_context = daemon.DaemonContext()
        daemon_context.working_directory = os.getcwd()
        daemon_context.files_preserve = [
            int(fd) for fd in os.listdir("/proc/self/fd")
            if int(fd) > 2]
        daemon_context.open()  # <- the magic happens here

    @staticmethod
    def _get_interactive_locals():
        """
        If we are running under IPython, extracts the local variables;
        otherwise, returns an empty dict.
        """
        try:
            __IPYTHON__  # pylint: disable=E0602
            from IPython.core.interactiveshell import InteractiveShell
            return {k: v for k, v in
                    InteractiveShell.instance().user_ns.items()
                    if k[0] != '_' and k not in (
                        "Out", "In", "exit", "quit", "get_ipython")}
        except NameError:
            return {}

    def _load_model(self, fname_workflow):
        self.info("Loading workflow \"%s\"...", fname_workflow)
        self.load_called = False
        self.main_called = False
        package_name, module_name = get_file_package_and_module(
            fname_workflow)
        try:
            return import_file_as_package(fname_workflow)
        except Exception as e:
            self.debug("Failed to import \"%s\" through the parent package "
                       "\"%s\": %s", fname_workflow, package_name, e)
            package_import_error = e
        # We failed to load the package => try module approach
        try:
            return import_file_as_module(fname_workflow)
        except FileNotFoundError:
            self.exception("Workflow does not exist: \"%s\"", fname_workflow)
            sys.exit(errno.ENOENT)
        except IsADirectoryError:
            self.exception("Workflow \"%s\" is a directory", fname_workflow)
            sys.exit(errno.EISDIR)
        except PermissionError:
            self.exception("Cannot read workflow \"%s\"", fname_workflow)
            sys.exit(errno.EACCES)
        except Exception as e:
            self.error("Failed to load the workflow \"%s\".\n"
                       "Package import error: %s\nModule import error: %s",
                       fname_workflow, package_import_error, e)
            sys.exit(Main.EXIT_FAILURE)

    def _apply_config(self, fname_config, config_list):
        self.info("Applying the configuration from %s...", fname_config)
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
        self.debug("Seeding with %s", rndvals)
        rndvals_split = rndvals.split(',')
        seeds = []
        for rndval, index in zip(rndvals_split, range(len(rndvals_split))):
            try:
                binvle = binascii.unhexlify(rndval)
                seed = numpy.frombuffer(binvle, dtype=numpy.uint8)
                prng.get(index + 1).seed(seed, dtype=numpy.uint8)
                seeds.append(seed)
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
                seeds.append(None)
                try:
                    prng.get(index + 1).seed(None)
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
                       count, dtype, fname, index + 1)
            try:
                seed = (numpy.fromfile(fname, dtype=dtype, count=count))
                prng.get(index + 1).seed(seed, dtype=dtype)
                seeds.append(seed)
            except:
                self.exception("Failed to seed the random generator with %s",
                               fname)
                sys.exit(Main.EXIT_FAILURE)
        self.seeds = seeds

    def _load_workflow(self, fname_snapshot):
        stype = splittype(fname_snapshot)[0]
        if stype == "odbc":
            import pyodbc

            addr = fname_snapshot[7:]
            parsed = addr.split('&')
            try:
                odbc, table, id_, log_id = parsed[:4]
            except TypeError:
                self.warning("Invalid ODBC source format. Here is the "
                             "template: odbc://<odbc data source spec>&"
                             "<table>&<id>&<log id>[&<optional name>]\n"
                             "<table> and <log id> may be empty (\"veles\" and"
                             " <id> value will be used).")
                return None
            if not table:
                table = "veles"
            if not log_id:
                log_id = id_
            if len(parsed) > 4:
                if len(parsed) > 5:
                    self.warning("Invalid ODBC source format")
                    return None
                name = parsed[-1]
            else:
                name = None
            try:
                return SnapshotterToDB.import_(odbc, table, id_, log_id, name)
            except pyodbc.Error as e:
                self.warning(
                    "Failed to load the snapshot from ODBC source: %s", e)
                return None
        elif stype in ("http", "https"):
            try:
                self.info("Downloading %s...", fname_snapshot)
                fname_snapshot = self.snapshot_file_name = wget.download(
                    fname_snapshot, root.common.snapshot_dir)
                print()
                sys.stdout.flush()
            except:
                self.exception("Failed to fetch the snapshot at \"%s\"",
                               fname_snapshot)
                return None
        try:
            return SnapshotterToFile.import_(fname_snapshot)
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
            self.launcher = Launcher(self.interactive)
            self.launcher.workflow_file = self.workflow_file
            self.launcher.config_file = self.config_file
            self.launcher.seeds = self.seeds
        except:
            self.exception("Failed to create the launcher")
            sys.exit(Main.EXIT_FAILURE)
        try:
            self.workflow = self._load_workflow(self.snapshot_file_name)
            self.snapshot = self.workflow is not None
            if not self.snapshot:
                wfkw = self._get_interactive_locals()
                wfkw.update(kwargs)
                self.workflow = Workflow(self.launcher, **wfkw)
                self.info("Created %s", self.workflow)
            else:
                self.info("Loaded the workflow snapshot from %s: %s",
                          self.snapshot_file_name, self.workflow)
                if self._visualization_mode:
                    self.workflow.plotters_are_enabled = True
                self.workflow.workflow = self.launcher
        except:
            self.exception("Failed to create the workflow")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)
        if self._workflow_graph:
            self.workflow.generate_graph(filename=self._workflow_graph,
                                         with_data_links=True,
                                         background='white')
        return self.workflow, self.snapshot

    def _main(self, **kwargs):
        if self._dry_run < 2:
            self.launcher.stop()
            return
        self.debug("main() was called from run()")
        if not self.load_called:
            self.critical("Call load() first in run()")
            sys.exit(Main.EXIT_FAILURE)
        self.main_called = True
        kwargs["snapshot"] = self.snapshot
        kwargs["no_device"] = not self.regular

        try:
            self.launcher.initialize(**kwargs)
        except:
            self.exception("Failed to initialize the launcher.")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)

        self.debug("Initialization is complete")
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
        finally:
            self.launcher.device_thread_pool_detach()

    def _dump_unit_attributes(self, arrays=True):
        import veles.external.prettytable as prettytable
        from veles.workflow import Workflow

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
                    if (not arrays and hasattr(v, "__len__") and len(v) > 32
                            and not isinstance(v, str)
                            and not isinstance(v, bytes)):
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

    def _run_core(self, wm):
        if self._dry_run <= 0:
            return
        if not self.optimization:
            from veles.genetics import fix_config
            fix_config(root)
        if self.regular:
            self.run_module(wm)
        elif self.optimization is not None:
            from veles.genetics import ConfigPopulation
            rand = prng.RandomGenerator(None)
            rand.state = prng.get().state
            ConfigPopulation(self, wm, rand=rand).evolve()
        elif self.ensemble_train is not None:
            import veles.ensemble.model_workflow as workflow
            self.run_module(workflow, model=wm, **self.ensemble_train.__dict__)
        elif self.ensemble_test is not None:
            import veles.ensemble.test_workflow as workflow
            self.run_module(workflow, model=wm, **self.ensemble_test.__dict__)
        else:
            raise NotImplementedError("Unsupported execution mode")

    def _apply_args(self, args):
        if args.background:
            self._daemonize()
        if not args.workflow:
            raise ValueError("Workflow path may not be empty")
        config_file = args.config
        if not config_file:
            raise ValueError("Configuration path may not be empty")
        if config_file == "-":
            config_file = "%s_config%s" % os.path.splitext(args.workflow)
        self.workflow_file = os.path.abspath(args.workflow)
        self.config_file = os.path.abspath(config_file)
        self._visualization_mode = args.visualize
        self._workflow_graph = args.workflow_graph
        self._dry_run = Main.DRY_RUN_CHOICES.index(args.dry_run)
        self._dump_attrs = args.dump_unit_attributes
        self.snapshot_file_name = args.snapshot
        self._parse_optimization(args)
        self._parse_ensemble_train(args)
        self._parse_ensemble_test(args)

    def _print_logo(self, args):
        if not args.no_logo:
            try:
                print(Main.LOGO)
            except:
                print(Main.LOGO.replace("©", "(c)"))
            sys.stdout.flush()

    def _print_version(self):
        print(veles.__version__, formatdate(veles.__date__, True),
              veles.__git__)

    def _print_config(self, cfg):
        io = StringIO()
        cfg.print_(file=io)
        self.debug("\n%s", io.getvalue().strip())

    def setup_logging(self, verbosity):
        try:
            super(Main, self).setup_logging(Main.LOG_LEVEL_MAP[verbosity])
        except Logger.LoggerHasBeenAlreadySetUp as e:
            if not self.interactive:
                raise from_none(e)

    def _register_print_max_rss(self):
        if not Main.registered_print_max_rss:
            atexit.register(self.print_max_rss)
            Main.registered_print_max_rss = True

    @staticmethod
    def format_decimal(val):
        if val < 1000:
            return str(val)
        d, m = divmod(val, 1000)
        return Main.format_decimal(d) + " %03d" % m

    def print_max_rss(self):
        res = resource.getrusage(resource.RUSAGE_SELF)
        if Watcher.max_mem_in_use > 0:
            self.info("Peak device memory used: %s Kb",
                      self.format_decimal(Watcher.max_mem_in_use // 1000))
        self.info("Peak resident memory used: %s Kb",
                  self.format_decimal(res.ru_maxrss))

    def run_module(self, module, **kwargs):
        self.debug("Calling %s.run()...", module.__name__)
        module.run(self._load, self._main, **kwargs)
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
        """Entry point for the VELES execution engine.
        """
        veles.validate_environment()

        ret = self._process_special_args()
        if ret is not None:
            return ret
        parser = Main.init_parser()
        args = parser.parse_args(self.argv)
        self._apply_args(args)

        self.setup_logging(args.verbosity)
        self._print_logo(args)
        for name in filter(str.strip, args.debug.split(',')):
            logging.getLogger(name).setLevel(logging.DEBUG)
        self._seed_random(args.random_seed)
        if args.debug_pickle:
            setup_pickle_debug()
        ThreadPool.reset()
        self._register_print_max_rss()

        if self.logger.isEnabledFor(logging.DEBUG):
            self._print_config(root)
        wm = self._load_model(self.workflow_file)
        self._apply_config(self.config_file, args.config_list)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._print_config(root)

        self._run_core(wm)

        if not self.interactive:
            self.info("End of job")
        else:
            self.info("\033[1;35mReturned the control\033[0m")
        return Main.EXIT_SUCCESS


def __run__():
    # Important: do not make these a one-liner! sys.exit is changed by
    # ThreadPool and in the case with merged call the original sys.exit will
    # be used instead, resulting in hanging on joining threads.
    retcode = Main().run()
    sys.exit(retcode)

if __name__ == "__main__":
    __run__()
