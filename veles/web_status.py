#!/usr/bin/python3

"""
Created on Feb 10, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
import logging
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
import tornado.escape
import tornado.ioloop as ioloop
import tornado.web as web
import uuid

from veles.config import root
import veles.external.daemon as daemon
import veles.logger as logger


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    PermissionError = IOError  # pylint: disable=W0622


debug_mode = False


class ServiceHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    @tornado.web.asynchronous
    def post(self):
        self.server.info("service POST from %s: %s", self.request.remote_ip,
                         self.request.body)
        try:
            data = tornado.escape.json_decode(self.request.body)
            self.server.send_command(self, data)
        except:
            self.server.exception()

    def finish_post(self, result):
        self.finish(result)


class UpdateHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        self.server.info("update POST from %s: %s", self.request.remote_ip,
                         self.request.body)
        try:
            data = tornado.escape.json_decode(self.request.body)
            self.server.receive_update(self.request.remote_ip, data)
        except:
            self.server.exception()


class WebStatusServer(logger.Logger):
    def __init__(self, cmd_queue_in, cmd_queue_out):
        super(WebStatusServer, self).__init__()
        self.application = web.Application([
            ("/service", ServiceHandler, {"server": self}),
            ("/" + root.common.web_status_update,
             UpdateHandler, {"server": self}),
            (r"/(js/.*)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            (r"/(css/.*)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            (r"/(fonts/.*)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            (r"/(img/.*)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            (r"/(.+\.html)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            ("/(veles.png)",
             web.StaticFileHandler, {'path': root.common.web_status_root}),
            ("/", web.RedirectHandler, {"url": "/status.html",
                                        "permanent": True}),
            ("", web.RedirectHandler, {"url": "/status.html",
                                       "permanent": True})
        ])
        self.application.listen(root.common.web_status_port)
        self.cmd_queue_in = cmd_queue_in
        self.cmd_queue_out = cmd_queue_out
        self.cmd_thread = threading.Thread(target=self.cmd_loop)
        self.pending_requests = {}
        self.info("HTTP server is going to listen on %s:%s",
                  socket.gethostname(), root.common.web_status_port)

    def send_command(self, handler, cmd):
        request_id = uuid.uuid4()
        self.pending_requests[request_id] = handler
        self.cmd_queue_out.put_nowait({"request": request_id, "body": cmd})

    def receive_update(self, addr, update):
        self.cmd_queue_out.put_nowait({"update": addr, "body": update})

    def cmd_loop(self):
        while True:
            cmd = self.cmd_queue_in.get()
            if cmd == "exit":
                self.cmd_queue_out.put_nowait(None)
                ioloop.IOLoop.instance().stop()
                break
            request_id = cmd["request"]
            self.pending_requests[request_id].finish_post(cmd["result"])
            del self.pending_requests[request_id]

    def run(self):
        self.info("HTTP server is running")
        self.cmd_thread.start()
        ioloop.IOLoop.instance().start()
        self.cmd_thread.join()


class WebStatus(logger.Logger):
    """
    Operates a web server based on Tornado to show various runtime information.

    TODO(v.markovtsev): when twisted.web.static is ported, we should rewrite.
    See:
    http://twistedmatrix.com/trac/ticket/6177
    https://twistedmatrix.com/documents/13.2.0/web/howto/using-twistedweb.html
    http://twistedmatrix.com/documents/current/web/howto/web-in-60/index.html
    """

    GARBAGE_TIMEOUT = 30

    @staticmethod
    def start_web_server(cmd_queue_in, cmd_queue_out):
        if not debug_mode:
            logger.Logger.redirect_all_logging_to_file(
                root.common.web_status_log_file, backups=9)
        try:
            WebStatusServer(cmd_queue_in, cmd_queue_out).run()
        except:
            cmd_queue_out.put_nowait(None)

    def __init__(self):
        super(WebStatus, self).__init__()
        self.exiting = False
        self.masters = {}
        self.cmd_queue_in = mp.Queue()
        self.cmd_queue_out = mp.Queue()
        self.cmd_thread = threading.Thread(target=self.cmd_loop)
        self.process = mp.Process(target=WebStatus.start_web_server,
                                  args=(self.cmd_queue_out, self.cmd_queue_in))
        self.info("Initialized")

    def run(self):
        self.cmd_thread.start()
        self.process.start()
        self.process.join()

    def stop(self):
        self.exiting = True
        self.cmd_queue_out.put("exit")
        self.cmd_thread.join()

    def cmd_loop(self):
        while not self.exiting:
            cmd = self.cmd_queue_in.get()
            if not cmd:
                break
            self.debug("New command %s", str(cmd))
            try:
                if "update" in cmd.keys():
                    mid = cmd["body"]["id"]
                    self.debug("Master %s yielded %s", mid, str(cmd["body"]))
                    self.masters[mid] = cmd["body"]
                    self.masters[mid]["last_update"] = time.time()
                elif "request" in cmd.keys():
                    if cmd["body"]["request"] == "workflows":
                        ret = {}
                        garbage = []
                        for mid, master in self.masters.items():
                            if (time.time() - master["last_update"] >
                                    WebStatus.GARBAGE_TIMEOUT):
                                garbage.append(mid)
                                continue
                            ret[mid] = {}
                            for item in cmd["body"]["args"]:
                                ret[mid][item] = master[item]
                        for mid in garbage:
                            self.info("Removing the garbage collected master %"
                                      "%s", mid)
                            del self.masters[mid]
                        self.debug("Request %s: %s", cmd["request"], str(ret))
                        self.cmd_queue_out.put_nowait(
                            {"request": cmd["request"], "result": ret})
                    else:
                        self.cmd_queue_out.put_nowait(
                            {"request": cmd["request"], "result": None})
                elif "stop" in cmd.keys():
                    self.info("Stopping everything")
                    self.stop()
            except:
                self.exception()


def main():
    WebStatus().run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False,
                        help="activates debugging mode (run in foreground, "
                        "DEBUG logging level)", action='store_true')
    args = parser.parse_args()
    debug_mode = args.debug
    if not debug_mode:
        pidfile = root.common.web_status_pidfile
        if not os.access(os.path.dirname(pidfile), os.W_OK):
            raise PermissionError(pidfile)
        print("Daemonizing, PID will be in ", pidfile + ".lock")
        with daemon.DaemonContext(pidfile=pidfile):
            print("Daemonized")
            logger.Logger.setup(level=logging.INFO)
            logger.Logger.redirect_all_logging_to_file(
                root.common.web_status_log_file, backups=9)
            main()
    else:
        logger.Logger.setup(level=logging.DEBUG)
        main()
