#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Feb 10, 2014

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
from bson import json_util
from collections import defaultdict
import logging
import json
import motor
import os
from six import print_
import socket
import sys
import time
from tornado.escape import json_decode
import tornado.gen as gen
from tornado.ioloop import IOLoop, PeriodicCallback
import tornado.web as web

from veles.compat import PermissionError, BrokenPipeError
from veles.config import root
import veles.external.daemon as daemon
from veles.logger import Logger


debug_mode = True


class ServiceHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    @web.asynchronous
    @gen.coroutine
    def post(self):
        self.server.debug("service POST from %s: %s", self.request.remote_ip,
                          self.request.body)
        try:
            data = json_decode(self.request.body)
            yield self.server.receive_request(self, data)
        except:
            self.server.exception("service POST")
            self.clear()
            self.finish({"request": data["request"] if data else "",
                         "result": "error"})


class UpdateHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    def post(self):
        self.server.debug("update POST from %s: %s", self.request.remote_ip,
                          self.request.body[:200] +
                          b"..." if len(self.request.body) > 200 else b"")
        try:
            data = json_decode(self.request.body)
            self.server.receive_update(self, data)
        except:
            self.server.exception("update POST")


class LogsHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    def get(self):
        session = self.get_argument("session", None)
        if session is None:
            self.clear()
            self.set_status(400)
        else:
            self.render("logs.html", session=session)


class WebServer(Logger):
    """
    Operates a web server based on Tornado to show various runtime information.
    """

    GARBAGE_TIMEOUT = 60

    def __init__(self, **kwargs):
        super(WebServer, self).__init__()
        if not debug_mode:
            Logger.redirect_all_logging_to_file(
                root.common.web.log_file, backups=root.common.web.log_backups)
        self.application = web.Application([
            ("/service", ServiceHandler, {"server": self}),
            ("/update", UpdateHandler, {"server": self}),
            ("/logs.html?.*", LogsHandler, {"server": self}),
            (r"/((js|css|fonts|img|maps)/.*)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            (r"/(.+\.html)",
             web.StaticFileHandler, {'path': root.common.web.root}),
            ("/", web.RedirectHandler, {"url": "/status.html",
                                        "permanent": True}),
            ("", web.RedirectHandler, {"url": "/status.html",
                                       "permanent": True})
        ], template_path=root.common.web.templates, gzip=not debug_mode)
        self._port = kwargs.get("port", root.common.web.port)
        self.application.listen(self._port)
        self.masters = {}
        self.motor = motor.MotorClient(
            "mongodb://" + kwargs.get("mongodb",
                                      root.common.mongodb_logging_address))
        self.db = self.motor.veles
        self.ensure_mongo_indexes()
        self.mongo_drop_time_threshold = kwargs.get(
            "mongo_drop_time_threshold", root.common.web.drop_time)
        self.mongo_dropper = PeriodicCallback(
            self.drop_old_mongo_records, self.mongo_drop_time_threshold * 1000)

    @property
    def port(self):
        return self._port

    def ensure_mongo_indexes(self):
        self.db.events.ensure_index(
            (("session", 1), ("instance", 1), ("name", 1), ("time", 1)))
        self.db.events.ensure_index((("session", 1), ("time", 1)))
        self.db.logs.ensure_index(
            (("session", 1), ("node", 1), ("levelname", 1)))
        self.db.logs.ensure_index((("session", 1), ("time", 1)))

    @gen.coroutine
    def drop_old_mongo_records(self):
        self.info("Discovering outdated MongoDB records...")
        logs_cursor = yield self.db.logs.aggregate(
            [{"$group": {"_id": "$session", "last": {"$max": "$created"}}}],
            cursor={})
        events_cursor = yield self.db.events.aggregate(
            [{"$group": {"_id": "$session", "last": {"$max": "$time"}}}],
            cursor={})
        to_remove = {"logs": [], "events": []}
        now = time.time()

        for cursor in (logs_cursor, events_cursor):
            while (yield cursor.fetch_next):
                obj = cursor.next_object()
                if obj['last'] is None:
                    self.warning("MongoDB aggregation failure for %s in %s",
                                 obj['_id'], cursor.collection.name)
                    continue
                if (obj['last'] - now) > self.mongo_drop_time_threshold:
                    to_remove[cursor.collection.name].append(obj['_id'])
            if len(to_remove[cursor.collection.name]) == 0:
                self.info("No outdated sessions was found in %s",
                          cursor.collection.name)

        for col in ("logs", "events"):
            if len(to_remove[col]) == 0:
                continue
            ack = yield self.db[col].remove(
                {"session": {"$in": to_remove[col]}})
            self.info("Removed %d sessions in %s with status %s",
                      ack["n"], col, "ok" if ack["ok"] else "error")

    @gen.coroutine
    def receive_request(self, handler, data):
        rtype = data["request"]
        if rtype == "workflows":
            ret = defaultdict(dict)
            garbage = []
            now = time.time()
            for mid, master in self.masters.items():
                if (now - master["last_update"] > WebServer.GARBAGE_TIMEOUT):
                    garbage.append(mid)
                    continue
                for item in data["args"]:
                    ret[mid][item] = master[item]
            for mid in garbage:
                self.info("Removing the garbage collected master %s", mid)
                del self.masters[mid]
            self.debug("Request %s: returning %d workflows", rtype, len(ret))
            handler.finish({"request": rtype, "result": ret})
        elif rtype in ("logs", "events"):
            query = data.get("find")
            if query is not None:
                cursor = self.db[rtype].find(query)
            else:
                query = data.get("aggregate")
                if query is not None:
                    cursor = yield self.db[rtype].aggregate(query, cursor={})
                else:
                    raise ValueError("Only 'find' and 'aggregate' commands are"
                                     " supported")
            handler.set_header("Content-Type",
                               "application/json; charset=UTF-8")
            handler.write("{\"request\": \"%s\", \"result\": [" % rtype)
            count = 0
            first = True
            while (yield cursor.fetch_next):
                if not first:
                    handler.write(",\n")
                else:
                    first = False
                json_raw = json.dumps(cursor.next_object(),
                                      default=json_util.default)
                handler.write(json_raw.replace("</", "<\\/"))
                count += 1
            handler.finish("]}")
            self.debug("Fetched %d \"%s\" documents", count, rtype)
        else:
            handler.finish({"request": rtype, "result": None})

    def receive_update(self, handler, data):
        mid = data["id"]
        graph = data["graph"]
        data["graph"] = "<cut>"
        self.debug("Master %s yielded %s", mid, data)
        data["graph"] = graph
        self.masters[mid] = data
        self.masters[mid]["last_update"] = time.time()

    def run(self):
        IOLoop.instance().add_callback(
            self.info, "HTTP server is running on %s:%s", socket.gethostname(),
            self.port)
        IOLoop.instance().add_callback(self.drop_old_mongo_records)
        self.mongo_dropper.start()
        IOLoop.instance().start()

    def stop(self):
        self.mongo_dropper.stop()
        IOLoop.instance().stop()


def main(**kwargs):
    WebServer(**kwargs).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False,
                        help="activates debugging mode (run in foreground, "
                        "DEBUG logging level)", action='store_true')
    parser.add_argument("-p", "--port", default=root.common.web.port, type=int,
                        help="Port to listen on.")
    parser.add_argument("-u", "--user", default=None,
                        help="Become this user (uid or name).")
    parser.add_argument("-g", "--group", default=None,
                        help="Belong to this group (gid or name).")
    args = parser.parse_args()
    debug_mode = args.debug
    if not debug_mode:
        pidfile = root.common.web.pidfile
        full_pidfile = pidfile + ".lock"
        if args.user is None and not os.access(os.path.dirname(full_pidfile),
                                               os.W_OK):
            raise PermissionError(pidfile)
        if os.path.exists(full_pidfile):
            real_pidfile = os.readlink(full_pidfile)
            pid = int(real_pidfile.split('.')[-1])
            print("Replacing PID %d..." % pid)
            try:
                os.kill(pid, 0)
            except OSError:
                os.remove(real_pidfile)
                os.remove(full_pidfile)
                print_("Detected a stale lock file %s" % real_pidfile,
                       file=sys.stderr)
            else:
                raise FileExistsError(full_pidfile)
        print("Daemonizing, PID will be referenced by ", full_pidfile)
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        with daemon.DaemonContext(pidfile=pidfile, stderr=sys.stderr,
                                  uid=args.user, gid=args.group):
            log_file = root.common.web.log_file
            Logger.setup_logging(level=logging.INFO)
            Logger.redirect_all_logging_to_file(log_file, backups=9)
            main(port=args.port)
    else:
        Logger.setup_logging(level=logging.DEBUG)
        main(port=args.port)
