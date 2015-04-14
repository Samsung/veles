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

Created on  Oct 12, 2014

Collaborative image labelling.

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


from collections import namedtuple
import hashlib
import json
import logging
from mimetypes import guess_type
import os
from PIL import Image
import pyinotify
import sys

from tornado.escape import json_decode
from tornado.ioloop import IOLoop
import tornado.gen as gen
from tornado.options import parse_command_line, define, options
import tornado.web as web

from veles.config import root


def json_file(file):
    return file + ".json"


class ThumbnailsHandler(web.StaticFileHandler):
    COMMON_PATH = "~/.thumbnails/normal"
    MODERN_PATH = "~/.cache/thumbnails/normal"

    def initialize(self):
        path = os.path.expanduser(ThumbnailsHandler.MODERN_PATH)
        if not os.path.exists(path):
            path = os.path.expanduser(ThumbnailsHandler.COMMON_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        super().initialize(path)
        self.logger = logging.getLogger("thumbnailer")

    @gen.coroutine
    def get(self, path, include_body=True):
        path = os.path.abspath(os.path.join(options.root,
                                            self.parse_url_path(path)))
        if not os.path.exists(path):
            raise web.HTTPError(404)
        file_hash = hashlib.md5(("file://" + path).encode("utf-8")).hexdigest()
        file_thumbnail = os.path.join(
            os.path.expanduser(ThumbnailsHandler.COMMON_PATH),
            file_hash) + ".png"
        if not os.path.exists(file_thumbnail) and os.path.exists(
                os.path.expanduser(ThumbnailsHandler.MODERN_PATH)):
            file_thumbnail = os.path.join(
                os.path.expanduser(ThumbnailsHandler.MODERN_PATH),
                file_hash) + ".png"
        if not os.path.exists(file_thumbnail):
            img = Image.open(path)
            img.thumbnail((128, 128), Image.ANTIALIAS)
            with open(file_thumbnail, "wb") as fout:
                img.save(fout, "PNG")
            self.logger.info(
                "generated %s for %s",
                os.path.join("~", os.path.relpath(file_thumbnail,
                                                  os.path.expanduser("~"))),
                os.path.relpath(path, options.root))
        yield super().get(os.path.basename(file_thumbnail), include_body)


class SelectionsHandler(web.RequestHandler):
    def post(self):
        data = json_decode(self.request.body)
        file_name = os.path.join(options.root, data["file"])
        selections = "[]"
        if os.access(json_file(file_name), os.R_OK):
            with open(json_file(file_name), "r") as fin:
                selections = fin.read()
        self.finish(selections)


class TouchedHandler(web.RequestHandler):
    def initialize(self, events_handler):
        self.touched = events_handler.touched

    def post(self):
        data = json_decode(self.request.body)
        response = {p: p in self.touched for p in data}
        self.finish(json.dumps(response))


class UpdateHandler(web.RequestHandler):
    def initialize(self):
        self.logger = logging.getLogger("update")

    def post(self):
        data = json_decode(self.request.body)
        file_name = os.path.join(options.root, data["file"])
        if os.path.exists(json_file(file_name)):
            if not data["overwrite"]:
                with open(json_file(file_name), "r") as fin:
                    sels = json.load(fin)
                if sels != data["selections"]:
                    raise web.HTTPError(403)
            else:
                self.logger.info("Overwriting %s", json_file(file_name))
        with open(json_file(file_name), "w") as fout:
            json.dump(data["selections"], fout)


class BBoxerHandler(web.RequestHandler):
    ImageTemplate = namedtuple(
        "ImageT", ("name", "path", "size", "format", "mode", "touched"))

    def initialize(self):
        self.logger = logging.getLogger("main")

    @staticmethod
    def discover_image(path):
        img = Image.open(path)
        return BBoxerHandler.ImageTemplate(
            name=os.path.splitext(os.path.basename(path))[0],
            path=os.path.relpath(path, options.root).replace("\\", "/"),
            size=img.size, format=img.format, mode=img.mode,
            touched=os.path.exists(json_file(path)))

    @staticmethod
    def is_image(path):
        mime = guess_type(path)[0]
        return (mime is not None and mime.startswith("image/") and
                mime.find("svg") < 0)

    @staticmethod
    def walk():
        app_root = os.path.abspath(os.path.dirname(__file__))
        for proot, dirs, files in os.walk(options.root):
            if os.path.abspath(proot).startswith(app_root):
                del dirs[:]
                continue
            for file in files:
                if BBoxerHandler.is_image(file):
                    yield os.path.join(proot, file)

    def get(self):
        images = []
        for path in BBoxerHandler.walk():
            try:
                images.append(BBoxerHandler.discover_image(path))
            except:
                self.logger.exception("Failed to load %s", path)
        return self.render("bboxer.html",
                           images=sorted(images, key=lambda i: i.path))


define("port", default=8080, type=int, help="Port which server should listen.")
define("root", default=".", help="Root directory to scan for images.")


class RootEventsNotifier(pyinotify.ProcessEvent):
    def __init__(self, logger):
        pyinotify.ProcessEvent.__init__(self)
        self.logger = logger
        self.touched = set((os.path.relpath(p, options.root)
                            for p in BBoxerHandler.walk()
                            if os.path.exists(json_file(p))))

    def process_IN_CREATE(self, event):
        if not event.pathname.endswith(".json"):
            return
        imgfile = event.pathname[:-5]
        if not os.path.exists(imgfile) or not BBoxerHandler.is_image(imgfile):
            return
        self.logger.info("%s was created",
                         os.path.relpath(event.pathname, options.root))
        key = os.path.relpath(imgfile, options.root)
        self.touched.add(key)

    def process_IN_DELETE(self, event):
        if not event.pathname.endswith(".json"):
            return
        imgfile = event.pathname[:-5]
        self.logger.info("%s was deleted",
                         os.path.relpath(event.pathname, options.root))
        key = os.path.relpath(imgfile, options.root)
        if key in self.touched:
            self.touched.remove(key)

    def process_IN_DELETE_SELF(self, event):
        self.logger.critical("%s no longer exists - exiting", options.root)
        IOLoop.instance().stop()


def main():
    parse_command_line()
    logger = logging.getLogger("bboxer")
    logger.info("Root is set to %s", os.path.abspath(options.root))
    # Add events watcher for options.root
    wm = pyinotify.WatchManager()
    handler = RootEventsNotifier(logger)
    notifier = pyinotify.TornadoAsyncNotifier(
        wm, IOLoop.instance(), default_proc_fun=handler)
    wm.add_watch(options.root, pyinotify.IN_CREATE | pyinotify.IN_DELETE |
                 pyinotify.IN_DELETE_SELF)
    app = web.Application([
        ("/bboxer.html", BBoxerHandler),
        ("/selections", SelectionsHandler),
        ("/update", UpdateHandler),
        ("/touched", TouchedHandler, {"events_handler": handler}),
        (r"/images/(?P<path>.*)",
         web.StaticFileHandler, {"path": options.root}),
        (r"/thumbnails/(?P<path>.*)", ThumbnailsHandler),
        (r"/((js|css|fonts|img|maps)/.*)",
         web.StaticFileHandler, {'path': root.common.web.root}),
        ("/", web.RedirectHandler,
         {"url": "/bboxer.html", "permanent": True}),
        ("", web.RedirectHandler,
         {"url": "/bboxer.html", "permanent": True})],
        template_path=root.common.web.templates
    )
    app.listen(options.port)
    logger.info("Listening on %d", options.port)
    try:
        IOLoop.instance().start()
    finally:
        notifier.stop()

if __name__ == "__main__":
    sys.exit(main())
