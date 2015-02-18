#!/usr/bin/python3
# encoding: utf-8
# PYTHON_ARGCOMPLETE_OK
"""
This is the only entry point of any VELES-based execution.

Contact:
    * g.kuznetsov@samsung.com
    * v.markovtsev@samsung.com

"""

import atexit
import binascii
from email.utils import formatdate
import errno
import gc
import logging
import numpy
import os
import resource
import runpy
from six import print_, StringIO
import sys

import veles
from veles.cmdline import CommandLineBase
from veles.compat import from_none
from veles.config import root
from veles.external import daemon
from veles.logger import Logger
from veles.launcher import Launcher
from veles.backends import Device
from veles.memory import Watcher
from veles.pickle2 import setup_pickle_debug
from veles import prng
from veles.snapshotter import Snapshotter
from veles.thread_pool import ThreadPool

if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622
    IsADirectoryError = IOError  # pylint: disable=W0622
    PermissionError = IOError  # pylint: disable=W0622


class Main(Logger, CommandLineBase):
    """
    Entry point of any VELES engine executions.
    """

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

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
    def acceleration_is_enabled(self):
        return not (self.launcher.is_master or self.optimization or
                    self._disable_acceleration)

    def _process_special_args(self):
        if len(sys.argv) > 1 and sys.argv[1] == "forge":
            from veles.forge_client import __run__ as forge_run
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
        if "--dump-config" in sys.argv:
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
        sys.argv[1:] = cmdline.split()
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
            (r"/((js|css|fonts|img)/.*)",
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
        package_name = os.path.basename(os.path.dirname(fname_workflow))
        module_name = os.path.splitext(os.path.basename(fname_workflow))[0]
        sys.path.insert(0, os.path.dirname(os.path.dirname(fname_workflow)))
        try:
            package = __import__("%s.%s" % (package_name, module_name))
            return getattr(package, module_name)
        except Exception as e:
            self.debug("Failed to import \"%s\" through the parent package "
                       "\"%s\": %s", package_name, e)
        finally:
            del sys.path[0]

        sys.path.insert(0, os.path.dirname(fname_workflow))
        try:
            return __import__(module_name)
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
        finally:
            del sys.path[0]

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
                prng.get(index + 1).seed(
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
                       count, str(dtype), fname, index + 1)
            try:
                prng.get(index + 1).seed(numpy.fromfile(
                    fname, dtype=dtype, count=count), dtype=dtype)
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
            self.device = Device() if self.acceleration_is_enabled else None
            if self.device is not None:
                self.device.thread_pool_attach(self.workflow.thread_pool)
        except:
            self.exception("Failed to create the OpenCL device.")
            self.launcher.stop()
            sys.exit(Main.EXIT_FAILURE)

        def device_thread_pool_detach():
            if self.device is not None:
                self.device.thread_pool_detach()

        try:
            self.workflow.initialize(device=self.device, **kwargs)
        except:
            self.exception("Failed to initialize the workflow")
            self.launcher.stop()
            device_thread_pool_detach()
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
        finally:
            device_thread_pool_detach()

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

    def _print_config(self, cfg):
        io = StringIO()
        cfg.print_(file=io)
        self.debug("\n%s", io.getvalue().strip())

    def print_max_rss(self):
        def space_group(val, first_pos=True):
            if val < 1000:
                return ("%d" if first_pos else "%03d") % val
            d, m = divmod(val, 1000)
            return space_group(d, d < 1000) + " %d" % m

        res = resource.getrusage(resource.RUSAGE_SELF)
        if Watcher.max_mem_in_use > 0:
            self.info("Peak device memory used: %s Kb",
                      space_group(Watcher.max_mem_in_use // 1000))
        self.info("Peak resident memory used: %s Kb",
                  space_group(res.ru_maxrss))

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
        self._disable_acceleration = args.disable_acceleration
        self._parse_optimization(args)

        Logger.setup(level=Main.LOG_LEVEL_MAP[args.verbosity])
        self._print_logo(args)
        for name in filter(str.strip, args.debug.split(',')):
            logging.getLogger(name).setLevel(logging.DEBUG)
        self._seed_random(args.random_seed)
        if args.debug_pickle:
            setup_pickle_debug()
        atexit.register(self.print_max_rss)

        if self.logger.isEnabledFor(logging.DEBUG):
            self._print_config(root)
        wm = self._load_model(fname_workflow, args.snapshot)
        self._apply_config(fname_config, args.config_list)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._print_config(root)

        self._run_core(wm, args.background)

        self.info("End of job")
        return Main.EXIT_SUCCESS


def __run__():
    # Important: do not make these a one-liner! sys.exit is changed by
    # ThreadPool and in the case with merged call the original sys.exit will
    # be used instead, resulting in hanging on joining threads.
    retcode = Main().run()
    sys.exit(retcode)

if __name__ == "__main__":
    __run__()
