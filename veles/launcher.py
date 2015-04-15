# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Feb 10, 2014

Workflow launcher (server/client/standalone).

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


import argparse
import datetime
import getpass
from itertools import chain
import json
import os
import paramiko
import platform
from six import BytesIO, add_metaclass
import socket
import sys
import threading
import time
from twisted.internet import reactor
from twisted.internet.error import ReactorNotRunning
from twisted.web.html import escape
from twisted.web.client import (Agent, HTTPConnectionPool, FileBodyProducer,
                                getPage)
from twisted.web.http_headers import Headers
import uuid

from veles.backends import Device, NumpyDevice
import veles.client as client
from veles.cmdline import CommandLineArgumentsRegistry, CommandLineBase
from veles.compat import from_none
from veles.config import root
import veles.graphics_server as graphics_server
from veles.plotter import Plotter
import veles.logger as logger
import veles.server as server
from veles.thread_pool import ThreadPool
from veles.error import MasterSlaveCommunicationError
from veles.external.pytrie import StringTrie


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622


def filter_argv(argv, *blacklist):
    ptree = StringTrie({v: i for i, v in enumerate(blacklist)})
    filtered = []
    maybe_value = False
    boolean_args = set(chain.from_iterable(
        a.option_strings for a in CommandLineBase.init_parser()._actions
        if isinstance(a, (argparse._StoreTrueAction,
                          argparse._StoreFalseAction))))
    for arg in argv:
        if maybe_value:
            maybe_value = False
            if not arg or arg[0] != "-":
                continue
        if ptree.longest_prefix(arg, None) is None:
            filtered.append(arg)
        else:
            maybe_value = arg not in boolean_args
    return filtered


@add_metaclass(CommandLineArgumentsRegistry)
class Launcher(logger.Logger):
    """Workflow launcher.

    Parameters:
        parser                A custom argparse.ArgumentParser instance.
        master_address        The server's address (implies Slave mode).
        listen_address        The address to listen (implies Master mode).
        matplotlib_backend    Matplotlib backend to use (only in Master mode).
        stealth               Do not report the status to the web server,
                              do not launch it if necessary (only in Master
                              mode).
        nodes                 The list of slaves to launch remotely (only in
                              Master mode).
        log_file              Duplicate all logging to this file.
    """

    graphics_client = None
    graphics_server = None

    def __init__(self, interactive=False, **kwargs):
        super(Launcher, self).__init__()
        parser = Launcher.init_parser(**kwargs)
        self.args, _ = parser.parse_known_args(self.argv)
        self.args.master_address = self.args.master_address.strip()
        self.args.listen_address = self.args.listen_address.strip()
        self.args.matplotlib_backend = self.args.matplotlib_backend.strip()
        self._slaves = [x.strip() for x in self.args.nodes.split(',')
                        if x.strip() != ""]
        self._slave_launch_transform = self.args.slave_launch_transform
        if self._slave_launch_transform.find("%s") < 0:
            raise ValueError("Slave launch command transform must contain %s")

        if self.args.log_file != "":
            log_file = self.args.log_file
            if self.args.log_file_pid:
                log_base_name = os.path.splitext(os.path.basename(log_file))
                log_file = os.path.join(
                    os.path.dirname(log_file),
                    "%s.%d%s" % (log_base_name[0], os.getpid(),
                                 log_base_name[1]))
            logger.Logger.redirect_all_logging_to_file(log_file)

        self.info("My Python is %s %s", platform.python_implementation(),
                  platform.python_version())
        self.info("My PID is %d", os.getpid())
        self.info("My time is %s", datetime.datetime.now())
        self.id = str(uuid.uuid4()) if not self.is_slave else None
        self.log_id = self.args.log_id or self.id
        if self.logs_to_mongo:
            if self.mongo_log_addr == "":
                self.args.log_mongo = root.common.mongodb_logging_address
            if not self.is_slave:
                logger.Logger.duplicate_all_logging_to_mongo(
                    self.args.log_mongo, self.log_id, "master")

        self._monkey_patch_twisted_failure()
        self._lock = threading.Lock()
        self._webagg_port = 0
        self._agent = None
        self._workflow = None
        self._initialized = False
        self._running = False
        self._start_time = None
        self._device = NumpyDevice()
        self._interactive = interactive
        self._reactor_thread = None
        self._notify_update_interval = kwargs.get(
            "status_update_interval",
            root.common.web.notification_interval)
        if self.args.yarn_nodes is not None and self.is_master:
            self._discover_nodes_from_yarn(self.args.yarn_nodes)

    def __getstate__(self):
        return {}

    def _monkey_patch_twisted_failure(self):
        from twisted.python.failure import Failure
        original_raise = Failure.raiseException
        launcher = self

        def raiseException(self):
            try:
                original_raise(self)
            except:
                launcher.exception("Error inside Twisted reactor:")
                reactor.callFromThread(launcher.stop)

        if original_raise != raiseException:
            Failure.raiseException = raiseException

    @staticmethod
    def init_parser(**kwargs):
        """
        Initializes an instance of argparse.ArgumentParser.
        """
        parser = kwargs.get("parser", argparse.ArgumentParser())
        parser.add_argument("-m", "--master-address", type=str,
                            default=kwargs.get("master_address", ""),
                            help="Workflow will be launched in client mode "
                            "and connected to the master at the specified "
                            "address.").mode = ["slave"]
        parser.add_argument("-l", "--listen-address", type=str,
                            default=kwargs.get("listen_address", ""),
                            help="Workflow will be launched in server mode "
                                 "and will accept client connections at the "
                                 "specified address.").mode = ["master"]
        parser.add_argument("-p", "--matplotlib-backend", type=str, nargs='?',
                            const="",
                            default=kwargs.get("matplotlib_backend",
                                               root.common.matplotlib_backend),
                            help="Matplotlib drawing backend.")
        parser.add_argument("--no-graphics-client",
                            default=kwargs.get("graphics_client", False),
                            help="Do not launch the graphics client. Server "
                            "will still be started unless matplotlib backend "
                            "is an empty string.", action='store_true')
        parser.add_argument("-s", "--stealth",
                            default=kwargs.get("stealth", False),
                            help="Do not report own status to the Web Status "
                                 "Server.",
                            action='store_true')
        parser.add_argument("-n", "--nodes", type=str,
                            default=kwargs.get("nodes", ""),
                            help="The list of slaves to launch remotely "
                                 "separated by commas. Slave format is "
                                 "host/OpenCLPlatformNumber:OpenCLDevice(s)xN,"
                                  "examples: host/0:0, host/1:0-2, "
                                  "host/0:2-3x3.").mode = ["master"]
        parser.add_argument("--validate-history",
                            default=False, help="Check the apply/generate "
                            "history on master.", action='store_true'
                            ).mode = ["master"]
        parser.add_argument("-f", "--log-file", type=str,
                            default=kwargs.get("log_file", ""),
                            help="The file name where logs will be copied.")
        parser.add_argument("--log-file-pid", default=False,
                            action='store_true',
                            help="Insert process ID into the log file name.")
        parser.add_argument("-g", "--log-mongo", type=str, nargs='?',
                            const="",
                            default=kwargs.get("log_mongo", "no"),
                            help="MongoDB server address where logs will be "
                                 "sent.")
        parser.add_argument("-i", "--log-id", type=str,
                            default=kwargs.get("log_id", ""),
                            help="Log identifier (used my Mongo logger).")
        parser.add_argument("--yarn-nodes", type=str, default=None,
                            help="Discover the nodes from this YARN "
                            "ResourceManager's address.").mode = ["master"]
        parser.add_argument("--max-nodes", type=int, default=0,
                            help="Max number of slaves launched. 0 means "
                            "unlimited number.").mode = ["master"]
        parser.add_argument("--slave-launch-transform", type=str, default="%s",
                            help="Transformation of the slave remote launch "
                            "command given over ssh (%%s corresponds to the "
                            "original command).").mode = ["master"]
        return parser

    @property
    def interactive(self):
        return self._interactive

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value
        if self.id is not None:
            self.info("My ID is %s", self.id)

    @property
    def log_id(self):
        return self._log_id

    @log_id.setter
    def log_id(self, value):
        self._log_id = value
        if self.log_id is not None:
            self.info("My log ID is %s", self.log_id)

    @property
    def runs_in_background(self):
        return self.args.background

    @property
    def logs_to_mongo(self):
        return self.args.log_mongo != "no"

    @property
    def mongo_log_addr(self):
        return self.args.log_mongo

    @property
    def matplotlib_backend(self):
        return self.args.matplotlib_backend

    @property
    def reports_web_status(self):
        return not self.args.stealth and not self.is_slave

    @property
    def slaves(self):
        return self._slaves if self.is_master else []

    @property
    def webagg_port(self):
        return self._webagg_port

    @property
    def is_master(self):
        return True if self.args.listen_address else False

    @property
    def is_slave(self):
        return True if self.args.master_address else False

    @property
    def is_standalone(self):
        return not self.is_master and not self.is_slave

    @property
    def mode(self):
        if self.is_master:
            return "master"
        if self.is_slave:
            return "slave"
        if self.is_standalone:
            return "standalone"
        raise RuntimeError("Impossible happened")

    @property
    def device(self):
        return self._device

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def is_running(self):
        return self._running

    @property
    def workflow(self):
        return self._workflow

    @property
    def agent(self):
        return self._agent

    @property
    def plots_endpoints(self):
        return (Launcher.graphics_server.endpoints["epgm"] +
                [Launcher.graphics_server.endpoints["ipc"]]) \
            if getattr(self, "graphics_server", None) is not None else []

    @property
    def start_time(self):
        return self._start_time

    def threadsafe(fn):
        def wrapped(self, *args, **kwargs):
            with self._lock:
                return fn(self, *args, **kwargs)
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_threadsafe'
        return wrapped

    @threadsafe
    def add_ref(self, workflow):
        """
        Links with the nested Workflow instance, so that we are able to
        initialize.
        """
        self._workflow = workflow
        workflow.run_is_blocking = False
        if self.is_slave or self.matplotlib_backend == "":
            workflow.plotters_are_enabled = False

    def del_ref(self, workflow):
        pass

    def on_workflow_finished(self):
        reactor.callFromThread(self.stop)

    def device_thread_pool_detach(self):
        if self.device.attached(self.workflow.thread_pool):
            self.device.thread_pool_detach(self.workflow.thread_pool)

    @threadsafe
    def initialize(self, snapshot=False, **kwargs):
        # Ensure reactor stops in some rare cases when it does not normally
        if not self.interactive:
            self.workflow.thread_pool.register_on_shutdown(
                Launcher._reactor_shutdown)
        else:
            self._interactive_shutdown_ref = self._interactive_shutdown
            ThreadPool.register_atexit(self._interactive_shutdown_ref)
        if self.is_slave:
            self._agent = client.Client(
                self.args.master_address, self.workflow)

            def on_id_received(node_id, log_id):
                self.id = node_id
                self.log_id = log_id
                if self.logs_to_mongo:
                    logger.Logger.duplicate_all_logging_to_mongo(
                        self.args.log_mongo, self.log_id, node_id)

            self.agent.on_id_received = on_id_received
        else:
            if self.reports_web_status:
                timeout = self._notify_update_interval / 2
                self._web_status_agent = Agent(
                    reactor, pool=HTTPConnectionPool(reactor),
                    connectTimeout=timeout)
                # Launch the status server if it's not been running yet
                self._launch_status()
            if self.workflow.plotters_are_enabled and \
                    (not self.interactive or Launcher.graphics_client is None):
                try:
                    Launcher.graphics_server, Launcher.graphics_client = \
                        graphics_server.GraphicsServer.launch(
                            self.workflow.thread_pool,
                            self.matplotlib_backend,
                            self._set_webagg_port,
                            self.args.no_graphics_client)
                except:
                    self.error("Failed to create the graphics server and/or "
                               "client. Try to completely disable plotting "
                               "with -p ''.")
                    raise
            elif self.args.no_graphics_client:
                self.warning("Plotters are disabled. --no-graphics-client has "
                             "no effect.")
            if self.is_master:
                try:
                    self._agent = server.Server(self.args.listen_address,
                                                self.workflow)
                    # Launch the nodes described in the command line or config
                    self._launch_nodes()
                except:
                    self._stop_graphics()
                    raise

        if not self.is_slave and self.reports_web_status:
            try:
                self.workflow_graph, _ = self.workflow.generate_graph(
                    filename=None, write_on_disk=False, with_data_links=True)
            except RuntimeError as e:
                self.warning("Failed to generate the workflow graph: %s", e)
                self.workflow_graph = ""

        try:
            if not self.is_master:
                self._device = Device()
        except Exception as e:
            self.error("Failed to create the OpenCL device.")
            raise from_none(e)

        self.workflow.reset_thread_pool()
        try:
            self.workflow.initialize(device=self.device, snapshot=snapshot,
                                     **kwargs)
        except Exception as e:
            self.error("Failed to initialize the workflow")
            self.device_thread_pool_detach()
            raise from_none(e)

        if not self.is_standalone:
            self._agent.initialize()
        if not self.interactive:
            trigger = reactor.addSystemEventTrigger
            trigger('before', 'shutdown', self._on_stop)
            trigger('after', 'shutdown', self._print_stats)
            trigger('after', 'shutdown', self.event, "work", "end", height=0.1)
        else:
            register = self.workflow.thread_pool.register_on_shutdown
            self._on_stop_ref = self._on_stop
            register(self._on_stop_ref)
            self._print_stats_ref = self._print_stats
            register(self._print_stats_ref)

            def work_end():
                self.event("work", "end", height=0.1)
            self._work_end = work_end
            register(self._work_end)
        for unit in self.workflow:
            if isinstance(unit, Plotter):
                unit.graphics_server = Launcher.graphics_server
        self._initialized = True

    def run(self):
        """Starts Twisted reactor, invokes attached workflow's run() and does
        periodic status updates.
        """
        self._pre_run()
        reactor.callLater(0, self.info, "Reactor is running")
        self.event("work", "begin", height=0.1)
        if self.interactive:
            if not reactor.running and self._reactor_thread is None:
                reactor._handleSignals()
                self._reactor_thread = threading.Thread(
                    name="TwistedReactor", target=reactor.run,
                    kwargs={"installSignalHandlers": False})
                self._reactor_thread.start()
            return
        try:
            reactor.run()
        except:
            self.exception("Reactor malfunction. The whole facility is going "
                           "to be destroyed in 10 minutes. Personnel "
                           "evacuation has been started.")
        finally:
            with self._lock:
                self._running = False

    def boot(self, **kwargs):
        """
        Initializes and runs the attached workflow.
        :param kwargs: The keyword arguments to pass to initialize().
        """
        self.initialize(**kwargs)
        self.run()

    def stop(self):
        """Stops Twisted reactor and Workflow execution.
        """
        with self._lock:
            if self.workflow is None:
                return
            running = self._running and reactor.running
            self._running = False
        if self.is_master and len(self.agent.protocols) > 0:
            self.info("Waiting for the slaves to finish (%d left)...",
                      len(self.agent.protocols))
            return
        if not running:
            self._on_stop()
            return
        if self.interactive:
            self._on_stop()
            return
        try:
            reactor.stop()
        except ReactorNotRunning:
            pass
        except:
            self.exception("Failed to stop the reactor. There is going to be "
                           "a meltdown unless you immediately activate the "
                           "emergency graphite protection.")

    def pause(self):
        self.workflow.thread_pool.pause()

    def resume(self):
        self.workflow.thread_pool.resume()

    @threadsafe
    def _pre_run(self):
        if not self._initialized:
            raise RuntimeError("Launcher was not initialized")
        if self._running:
            raise RuntimeError("Launcher is already running")
        self._running = True
        self._start_time = time.time()
        if self.reports_web_status:
            self._notify_update_last_time = self.start_time
            self._notify_status()
        if not self.is_slave:
            self.workflow.thread_pool.start()
            self.workflow.thread_pool.callInThread(self.workflow.run)

    def _on_stop(self):
        if self.workflow is None or not self._initialized:
            return
        self._on_stop_locked()

    @threadsafe
    def _on_stop_locked(self):
        self.info("Stopping everything (%s mode)", self.mode)
        self._initialized = False
        self._running = False
        # Wait for the own graphics client to terminate normally
        self._stop_graphics()
        self.workflow.thread_pool.shutdown()
        if self.args.validate_history:
            try:
                self.workflow.validate_history()
            except MasterSlaveCommunicationError as e:
                self.error("Workflow history validation was failed: %s", e)

    threadsafe = staticmethod(threadsafe)

    @staticmethod
    def _prepare_reactor_shutdown():
        original_stop = reactor.stop

        def stop():
            try:
                original_stop()
            except ReactorNotRunning:
                pass

        reactor.stop = stop

    @staticmethod
    def _reactor_shutdown():
        Launcher._prepare_reactor_shutdown()
        reactor.sigInt()

    def _interactive_shutdown(self):
        assert self.interactive
        self.debug("Shutting down in interactive mode")
        Launcher._prepare_reactor_shutdown()
        reactor.callFromThread(reactor.stop)
        self._stop_graphics(True)
        if self._reactor_thread is not None and \
                self._reactor_thread.is_alive():
            self._reactor_thread.join()

    def _stop_graphics(self, interactive_stop=False):
        if self.interactive and not interactive_stop:
            return
        if Launcher.graphics_client is not None:
            attempt = 0
            while Launcher.graphics_client.poll() is None and attempt < 10:
                Launcher.graphics_server.shutdown()
                attempt += 1
                time.sleep(0.2)
            if Launcher.graphics_client.poll() is None:
                Launcher.graphics_client.terminate()
                self.info("Graphics client has been terminated")
            else:
                self.info("Graphics client returned normally")

    def _print_stats(self):
        self.workflow.print_stats()
        if self.agent is not None:
            self.agent.print_stats()
        if self.start_time is not None:
            self.info(
                "Time elapsed: %s", datetime.timedelta(
                    seconds=(time.time() - self.start_time)))

    def _launch_status(self):
        if not self.reports_web_status:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((root.common.web.host,
                                  root.common.web.port))
        sock.close()
        if result != 0:
            self.info("Launching the web status server")
            self._launch_remote_progs(
                root.common.web.host,
                "PYTHONPATH=%s %s 2>>%s" %
                (os.path.dirname(
                    os.path.abspath(
                        os.path.join(root.common.veles_dir, "veles"))),
                 os.path.abspath(os.path.join(root.common.veles_dir,
                                              "veles/web_status.py")),
                 "%s.stderr%s" %
                 os.path.splitext(root.common.web.log_file)))
        else:
            self.info("Web status server %s:%d is already running",
                      root.common.web.host, root.common.web.port)

    def _launch_nodes(self):
        if len(self.slaves) == 0:
            return
        self.debug("Will launch the following slaves: %s",
                   ', '.join(self.slaves))
        filtered_argv = filter_argv(
            sys.argv, "-l", "--listen-address", "-n", "--nodes", "-p",
            "--matplotlib-backend", "-b", "--background", "-s", "--stealth",
            "-d", "--device", "--slave-launch-transform")[1:]
        host = self.args.listen_address[0:self.args.listen_address.index(':')]
        port = self.args.listen_address[len(host) + 1:]
        # No way we can send 'localhost' or empty host name to a slave.
        if not host or host == "0.0.0.0" or host == "localhost" or \
           host == "127.0.0.1":
            host = socket.gethostname()
        filtered_argv.insert(0, "-m %s:%s -b -i \"%s\"" % (host, port,
                                                           self.log_id))
        slave_args = " ".join(filtered_argv)
        self.debug("Slave args: %s", slave_args)
        total_slaves = 0
        max_slaves = self.args.max_nodes or 1000
        cmdline = "%s %s" % (sys.executable, os.path.abspath(sys.argv[0])) + \
            " --backend %s --device %s " + slave_args
        if self.args.log_file:
            cmdline += " &>> " + self.args.log_file
        for node in self.slaves:
            host, devs = node.split('/')
            progs = [cmdline % dev for dev in Device.iterparse(devs)]
            if total_slaves + len(progs) > max_slaves:
                progs = progs[:max_slaves - total_slaves]
            total_slaves += len(progs)
            self._launch_remote_progs(host, *progs)
            if total_slaves >= max_slaves:
                break

    def _launch_remote_progs(self, host, *progs):
        self.info("Launching %d instance(s) on %s", len(progs), host)
        cwd = os.getcwd()
        self.debug("cwd: %s", os.getcwd())
        PYTHONPATH = os.getenv("PYTHONPATH")
        if os.path.splitext(os.path.basename(sys.argv[0]))[0] == "__main__":
            if PYTHONPATH is None:
                PYTHONPATH = cwd
            else:
                PYTHONPATH += ":" + cwd
        if PYTHONPATH is not None:
            self.debug("PYTHONPATH: %s", PYTHONPATH)
            ppenv = "export PYTHONPATH='%s' && " % PYTHONPATH
        else:
            ppenv = ""
        pc = paramiko.SSHClient()
        try:
            pc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                pc.connect(host, look_for_keys=True, timeout=0.2)
            except paramiko.ssh_exception.SSHException:
                self.exception("Failed to connect to %s", host)
                return
            buf_size = 128
            channel = pc.get_transport().open_session()
            channel.get_pty()
            for prog in progs:
                prog = prog.replace(r'"', r'\"').replace(r"'", r"\'")
                cmd = self._slave_launch_transform % ("cd '%s' && %s%s" %
                                                      (cwd, ppenv, prog))
                self.debug("Executing %s", cmd)
                channel.exec_command(cmd)
                answer = channel.recv(buf_size)
                if answer:
                    self.warning("SSH returned %s", answer)
                    if len(answer) == buf_size:
                        self.warning("SSH answer looks longer than %d bytes, "
                                     "draining the stream...", buf_size)
                        while channel.recv(buf_size):
                            pass
            channel.close()
        except:
            self.exception("Failed to launch '%s' on %s", progs, host)
        finally:
            pc.close()

    def _set_webagg_port(self, port):
        self.info("Found out the WebAgg port: %d", port)
        self._webagg_port = port

    def _on_notify_status_error(self, error):
        self.warning("Failed to upload the status: %s", error)
        reactor.callLater(self._notify_update_interval, self._notify_status)

    def _notify_status(self, response=None):
        if not self._running:
            return
        time_passed = time.time() - self._notify_update_last_time
        if time_passed < self._notify_update_interval:
            reactor.callLater(self._notify_update_interval - time_passed,
                              self._notify_status)
            return
        self._notify_update_last_time = time.time()
        mins, secs = divmod(time.time() - self.start_time, 60)
        hours, mins = divmod(mins, 60)
        ret = {'id': self.id,
               'log_id': self.log_id,
               'name': self.workflow.name,
               'master': socket.gethostname(),
               'time': "%02d:%02d:%02d" % (hours, mins, secs),
               'user': getpass.getuser(),
               'graph': self.workflow_graph,
               'log_addr': self.mongo_log_addr,
               'slaves': self._agent.nodes if self.is_master else [],
               'plots': "http://%s:%d" % (socket.gethostname(),
                                          self.webagg_port),
               'custom_plots': "<br/>".join(self.plots_endpoints),
               'description':
               "<br />".join(escape(self.workflow.__doc__).split("\n"))}
        url = "http://%s:%d/update" % (root.common.web.host,
                                       root.common.web.port)
        headers = Headers({b'User-Agent': [b'twisted']})
        body = FileBodyProducer(BytesIO(json.dumps(ret).encode('charmap')))
        self.debug("Uploading status update to %s", url)
        d = self._web_status_agent.request(
            b'POST', url.encode('ascii'), headers=headers, bodyProducer=body)
        d.addCallback(self._notify_status)
        d.addErrback(self._on_notify_status_error)

    def _discover_nodes_from_yarn(self, address):
        if address.find(':') < 0:
            address += ":8088"
        if address[:7] != "http://":
            address = "http://" + address
        address += "/ws/v1/cluster/nodes"
        self.debug("Requesting GET %s", address)
        getPage(address.encode('ascii')).addCallbacks(
            callback=self._parse_yarn_nodes_json,
            errback=lambda error: self.warning(
                "Failed to get the nodes list from YARN ResourceManager: %s",
                error))

    def _parse_yarn_nodes_json(self, response):
        rstr = response.decode()
        self.debug("Received YARN response: %s", rstr)
        tree = json.loads(rstr)
        for node in tree["nodes"]["node"]:
            self._slaves.append(node["nodeHostName"] + "/0:0")
        reactor.callLater(0, self._launch_nodes)
