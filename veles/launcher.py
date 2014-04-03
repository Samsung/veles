"""
Created on Feb 10, 2014

Workflow launcher (server/client/standalone).

@author: Kazantsev Alexey <a.kazantsev@samsung.com>,
         Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import argparse
import daemon
import datetime
import getpass
import json
import os
import paramiko
import socket
import sys
import threading
import time
from tornado.ioloop import IOLoop
from tornado.httpclient import AsyncHTTPClient
from twisted.internet import reactor, threads, task
from twisted.web.html import escape
import uuid

import veles.client as client
from veles.config import root
import veles.graphics_server as graphics_server
import veles.logger as logger
import veles.server as server


def threadsafe(fn):
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapped


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
        self.info("My log ID is %s", self.log_id)
        self._lock = threading.Lock()
        self._webagg_port = 0
        self._agent = None
        self._workflow = None
        self._initialized = False
        self._running = False
        self._start_time = None

    def __getstate__(self):
        return {}

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
        if self.is_slave or self.matplotlib_backend == "":
            workflow.plotters_are_enabled = False
        if self.is_slave:
            self._agent = client.Client(self.args.master_address, workflow)
        else:
            if self.reports_web_status:
                self.tornado_ioloop_thread = threading.Thread(
                    target=IOLoop.instance().start)
                self._notify_task = task.LoopingCall(self._notify_status)
                self._notify_agent = AsyncHTTPClient()
                # Launch the status server if it's not been running yet
                self._launch_status()
            if workflow.plotters_are_enabled:
                self.graphics_server, self.graphics_client = \
                    graphics_server.GraphicsServer.launch_pair(
                        workflow.thread_pool, self.matplotlib_backend,
                        self._set_webagg_port)
            if self.is_master:
                self._agent = server.Server(self.args.listen_address, workflow)
                # Launch the nodes described in the configuration file/string
                self._launch_nodes()
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

    @threadsafe
    def stop(self):
        if not self._initialized:
            return
        if not self._running:
            self._on_stop()
            return
        self.info("Stopping everything (%s mode)", self.mode)
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
        reactor.addSystemEventTrigger('before', 'shutdown', self._on_stop)
        reactor.addSystemEventTrigger('after', 'shutdown', self._print_stats)
        self._start_time = time.time()
        if not self.is_slave:
            self.workflow_graph = self.workflow.generate_graph(
                write_on_disk=False)
        if self.reports_web_status:
            self.start_time = time.time()
            self.tornado_ioloop_thread.start()
            self._notify_task.start(
                root.common.web_status_notification_interval, now=False)
        if not self.is_slave:
            threads.deferToThreadPool(reactor, self.workflow.thread_pool,
                                      self.workflow.run)

    def _on_stop(self):
        self._initialized = False
        self._running = False
        # Kill the Web status Server notification task and thread
        if self.reports_web_status:
            self._notify_task.stop()
            IOLoop.instance().stop()
            self.tornado_ioloop_thread.join()
        # Wait for the own graphics client to terminate normally
        if self.workflow.plotters_are_enabled:
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
        self.workflow.thread_pool.shutdown()

    def _print_stats(self):
        self.workflow.print_stats()
        if self._start_time is not None:
            self.info("Time elapsed: %s",
                      str(datetime.timedelta(
                          seconds=(time.time() - self._start_time))))

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
                "PYTHONPATH=%s %s" %
                (os.path.dirname(
                    os.path.abspath(
                        os.path.join(root.common.veles_dir, "veles"))),
                 os.path.abspath(os.path.join(root.common.veles_dir,
                                              "veles/web_status.py"))))
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
        for node in self.slaves:
            host, devs = node.split('/')
            marray = devs.split('x')
            if len(marray) > 1:
                multiplier = int(marray[1])
            else:
                multiplier = 1
            oclpnums, ocldevnum = marray[0].split(':')
            oclpnum = int(oclpnums)
            ocldevarr = ocldevnum.split('-')
            ocldevmin = int(ocldevarr[0])
            if len(ocldevarr) > 1:
                ocldevmax = int(ocldevarr[1])
            else:
                ocldevmax = ocldevmin
            progs = []
            for _ in range(0, multiplier):
                for d in range(ocldevmin, ocldevmax + 1):
                    progs.append("%s %s -d %d:%d" % (
                                     os.path.abspath(sys.argv[0]),
                                     slave_args, oclpnum, d))
            self._launch_remote_progs(host, *progs)

    def _launch_remote_progs(self, host, *progs):
        self.info("Launching %d instance(s) on %s", len(progs), host)
        cwd = os.getcwd()
        self.debug("cwd: %s", os.getcwd())
        client = paramiko.SSHClient()
        try:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, look_for_keys=True, timeout=0.1)
            for prog in progs:
                self.debug("Launching %s", prog)
                client.exec_command("cd '%s' && %s" % (cwd, prog))
        except:
            self.exception()
        finally:
            client.close()

    def _set_webagg_port(self, port):
        self.info("Found out the WebAgg port: %d", port)
        self._webagg_port = port

    def _handle_notify_request(self, response):
        if response.error:
            self.warning(
                "Failed to upload the status update to %s:%s",
                root.common.web_status_host, root.common.web_status_port)
        else:
            self.debug("Successfully updated the status")

    def _notify_status(self):
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
        timeout = root.common.web_status_notification_interval / 2
        self._notify_agent.fetch("http://%s:%d/%s" %
                                 (root.common.web_status_host,
                                  root.common.web_status_port,
                                  root.common.web_status_update),
                                 self._handle_notify_request,
                                 method='POST', headers=None,
                                 connect_timeout=timeout,
                                 request_timeout=timeout,
                                 body=json.dumps(ret))
