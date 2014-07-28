"""
Created on Feb 10, 2014

Workflow launcher (server/client/standalone).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
         Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import argparse
import veles.external.daemon as daemon
import datetime
import getpass
import json
import os
import paramiko
from six import BytesIO, add_metaclass
import socket
import sys
import threading
import time
from twisted.internet import reactor
from twisted.web.html import escape
from twisted.web.client import (Agent, HTTPConnectionPool, FileBodyProducer,
                                getPage)
from twisted.web.http_headers import Headers
import uuid

import veles.client as client
from veles.cmdline import CommandLineArgumentsRegistry
from veles.config import root
import veles.graphics_server as graphics_server
import veles.logger as logger
import veles.server as server
from veles.error import MasterSlaveCommunicationError


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622


def threadsafe(fn):
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
    wrapped.__name__ = name + '_threadsafe'
    return wrapped


@add_metaclass(CommandLineArgumentsRegistry)
class Launcher(logger.Logger):
    """Workflow launcher.

    Parameters:
        parser                A custom argparse.ArgumentParser instance.
        master_address        The server's address (implies Slave mode).
        listen_address        The address to listen (implies Master mode).
        matplotlib_backend    Matplotlib backend to use (only in Master mode).
        background            Run in background as a daemon.
        stealth               Do not report the status to the web server,
                              do not launch it if necessary (only in Master
                              mode).
        nodes                 The list of slaves to launch remotely (only in
                              Master mode).
        log_file              Duplicate all logging to this file.
    """

    def __init__(self, **kwargs):
        super(Launcher, self).__init__()
        parser = Launcher.init_parser(**kwargs)
        self.args, _ = parser.parse_known_args()
        self.args.master_address = self.args.master_address.strip()
        self.args.listen_address = self.args.listen_address.strip()
        self.args.matplotlib_backend = self.args.matplotlib_backend.strip()
        self._slaves = [x.strip() for x in self.args.nodes.split(',')
                        if x.strip() != ""]
        if self.runs_in_background:
            self._daemon_context = daemon.DaemonContext()
            self._daemon_context.working_directory = os.getcwd()
            twisted_epollfd = None
            for fd in os.listdir("/proc/self/fd"):
                try:
                    if os.readlink("/proc/self/fd/" + fd) == \
                       "anon_inode:[eventpoll]":
                        twisted_epollfd = int(fd)
                except FileNotFoundError:
                    pass
            if twisted_epollfd is None:
                raise RuntimeError("Twisted reactor was not imported")
            self._daemon_context.files_preserve = list(range(
                twisted_epollfd, twisted_epollfd + 3))
            self.info("Daemonized")
            self._daemon_context.open()
        if self.args.log_file != "":
            logger.Logger.redirect_all_logging_to_file(self.args.log_file)

        self._id = str(uuid.uuid4())
        self._log_id = self.args.log_id or self.id
        if self.logs_to_mongo:
            if self.mongo_log_addr == "":
                self.args.log_mongo = root.common.mongodb_logging_address
            logger.Logger.duplicate_all_logging_to_mongo(self.args.log_mongo,
                                                         self._log_id)
        self._monkey_patch_twisted_failure()
        self.info("My PID is %d", os.getpid())
        self.info("My log ID is %s", self.log_id)
        self._lock = threading.Lock()
        self._webagg_port = 0
        self._agent = None
        self._workflow = None
        self._initialized = False
        self._running = False
        self._start_time = None
        self.graphics_client = None
        self._notify_update_interval = kwargs.get(
            "status_update_interval",
            root.common.web_status_notification_interval)
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
                launcher.stop()

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
                            "address.")
        parser.add_argument("-l", "--listen-address", type=str,
                            default=kwargs.get("listen_address", ""),
                            help="Workflow will be launched in server mode "
                                 "and will accept client connections at the "
                                 "specified address.")
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
        parser.add_argument("-b", "--background",
                            default=kwargs.get("background", False),
                            help="Run in background as a daemon.",
                            action='store_true')
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
                                  "host/0:2-3x3.")
        parser.add_argument("--validate-history",
                            default=False, help="Check the apply/generate "
                            "history on master.", action='store_true')
        parser.add_argument("-f", "--log-file", type=str,
                            default=kwargs.get("log_file", ""),
                            help="The file name where logs will be copied.")
        parser.add_argument("-g", "--log-mongo", type=str, nargs='?',
                            const="",
                            default=kwargs.get("log_mongo", "no"),
                            help="Mongo ZMQ endpoint where logs will be "
                                 "copied.")
        parser.add_argument("-i", "--log-id", type=str,
                            default=kwargs.get("log_id", ""),
                            help="Log identifier (used my Mongo logger).")
        parser.add_argument("--yarn-nodes", type=str, default=None,
                            help="Discover the nodes from this YARN "
                            "ResourceManager's address.")
        parser.add_argument("--max-nodes", type=int, default=0,
                            help="Max number of slaves launched. 0 means "
                            "unlimited number.")
        return parser

    @property
    def id(self):
        return self._id

    @property
    def log_id(self):
        return self._log_id

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
        return self.graphics_server.endpoints["epgm"] + \
            [self.graphics_server.endpoints["ipc"]] \
            if hasattr(self, "graphics_server") else []

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

        def shutdown():
            reactor.callLater(0, reactor.sigInt)

        # Ensure reactor stops in some rare cases when it does not normally
        self.workflow.thread_pool.register_on_shutdown(shutdown)
        if self.is_slave:
            self._agent = client.Client(self.args.master_address, workflow)
        else:
            if self.reports_web_status:
                timeout = self._notify_update_interval / 2
                self._web_status_agent = Agent(
                    reactor, pool=HTTPConnectionPool(reactor),
                    connectTimeout=timeout)
                # Launch the status server if it's not been running yet
                self._launch_status()
            if workflow.plotters_are_enabled:
                self.graphics_server, self.graphics_client = \
                    graphics_server.GraphicsServer.launch(
                        workflow.thread_pool, self.matplotlib_backend,
                        self._set_webagg_port, self.args.no_graphics_client)
            if self.is_master:
                try:
                    self._agent = server.Server(self.args.listen_address,
                                                workflow)
                    # Launch the nodes described in the command line or config
                    self._launch_nodes()
                except:
                    self._stop_graphics()
                    raise

        self._initialized = True

    def del_ref(self, workflow):
        pass

    def on_workflow_finished(self):
        reactor.callFromThread(self.stop)

    def run(self):
        self._pre_run()
        reactor.callLater(0, self.info, "Reactor is running")
        try:
            reactor.run()
        except:
            self.exception("Reactor malfunction. The whole facility is going "
                           "to be destroyed in 10 minutes. Personnel "
                           "evacuation has been started.")
        finally:
            with self._lock:
                self._running = False

    def stop(self):
        with self._lock:
            if not self._initialized:
                return
            running = self._running and reactor.running
        self.info("Stopping everything (%s mode)", self.mode)
        if not running:
            self._on_stop()
            return
        try:
            reactor.stop()
        except:
            self.exception("Failed to stop the reactor. There is going to be "
                           "a meltdown unless you immediately activate the "
                           "emergency graphite protection.")

    @threadsafe
    def _pre_run(self):
        if not self._initialized:
            raise RuntimeError("Launcher was not initialized")
        self._running = True
        if not self.is_standalone:
            self._agent.initialize()
        reactor.addSystemEventTrigger('before', 'shutdown', self._on_stop)
        reactor.addSystemEventTrigger('after', 'shutdown', self._print_stats)
        self._start_time = time.time()
        if not self.is_slave:
            self.workflow_graph, _ = self.workflow.generate_graph(
                filename=None, write_on_disk=False)
        if self.reports_web_status:
            self.start_time = time.time()
            self._notify_update_last_time = self.start_time
            self._notify_status()
        if not self.is_slave:
            self.workflow.thread_pool.callInThread(self.workflow.run)

    @threadsafe
    def _on_stop(self):
        if not self._initialized:
            return
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

    def _stop_graphics(self):
        if self.graphics_client is not None:
            attempt = 0
            while self.graphics_client.poll() is None and attempt < 10:
                self.graphics_server.shutdown()
                attempt += 1
                time.sleep(0.2)
            if self.graphics_client.poll() is None:
                self.graphics_client.terminate()
                self.info("Graphics client has been terminated")
            else:
                self.info("Graphics client returned normally")

    def _print_stats(self):
        self.workflow.print_stats()
        if self._start_time is not None:
            self.info(
                "Time elapsed: %s", datetime.timedelta(
                    seconds=(time.time() - self._start_time)))

    def _launch_status(self):
        if not self.reports_web_status:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((root.common.web_status_host,
                                  root.common.web_status_port))
        sock.close()
        if result != 0:
            self.info("Launching the web status server")
            self._launch_remote_progs(
                root.common.web_status_host,
                "PYTHONPATH=%s %s 2>>%s" %
                (os.path.dirname(
                    os.path.abspath(
                        os.path.join(root.common.veles_dir, "veles"))),
                 os.path.abspath(os.path.join(root.common.veles_dir,
                                              "veles/web_status.py")),
                 "%s.stderr%s" %
                 os.path.splitext(root.common.web_status_log_file)))
        else:
            self.info("Discovered an already running web status server")

    def _launch_nodes(self):
        if len(self.slaves) == 0:
            return
        self.debug("Will launch the following slaves: %s",
                   ', '.join(self.slaves))
        filtered_argv = []
        skip = False
        ignored_args = {"-l", "--listen-address", "-n", "--nodes", "-p",
                        "--matplotlib-backend", "-b", "--background",
                        "-s", "--stealth", "-d", "--device"}
        for i in range(1, len(sys.argv)):
            if sys.argv[i] in ignored_args:
                skip = True
            elif not skip:
                filtered_argv.append(sys.argv[i])
            else:
                skip = False
        filtered_argv.append("-m")
        host = self.args.listen_address[0:self.args.listen_address.index(':')]
        port = self.args.listen_address[len(host) + 1:]
        # No way we can send 'localhost' or empty host name to a slave.
        if not host or host == "0.0.0.0" or host == "localhost" or \
           host == "127.0.0.1":
            host = socket.gethostname()
        filtered_argv.append("%s:%s" % (host, port))
        filtered_argv.append("-b")
        filtered_argv.append("-i %s" % self.log_id)
        slave_args = " ".join(filtered_argv)
        self.debug("Slave args: %s", slave_args)
        total_slaves = 0
        max_slaves = self.args.max_nodes or 1000
        cmdline = "%s %s -d %d:%d"
        if self.args.log_file:
            cmdline += " &>> " + self.args.log_file
        for node in self.slaves:
            host, devs = node.split('/')
            marray = devs.split('x')
            multiplier = int(marray[1]) if len(marray) > 1 else 1
            oclpnums, ocldevnum = marray[0].split(':')
            oclpnum = int(oclpnums)
            ocldevarr = ocldevnum.split('-')
            ocldevmin = int(ocldevarr[0])
            if len(ocldevarr) > 1:
                ocldevmax = int(ocldevarr[1])
            else:
                ocldevmax = ocldevmin
            progs = []
            for _ in range(multiplier):
                for d in range(ocldevmin, ocldevmax + 1):
                    progs.append(cmdline % (os.path.abspath(sys.argv[0]),
                                            slave_args, oclpnum, d))
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
        pc = paramiko.SSHClient()
        try:
            pc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                pc.connect(host, look_for_keys=True, timeout=0.2)
            except paramiko.ssh_exception.SSHException:
                self.exception("Failed to connect to %s", host)
                return
            for prog in progs:
                prog = prog.replace(r'"', r'\"').replace(r"'", r"\'")
                self.debug("Launching %s", prog)
                pc.exec_command("cd '%s' && %s" % (cwd, prog))
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
        url = "http://%s:%d/%s" % (root.common.web_status_host,
                                   root.common.web_status_port,
                                   root.common.web_status_update)
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
