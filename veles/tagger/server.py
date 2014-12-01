#!/usr/bin/env python3
from collections import namedtuple

import hashlib
import json
import logging
from mimetypes import guess_type
import os
from PIL import Image
import sys

from tornado.escape import json_decode
from tornado.ioloop import IOLoop
import tornado.gen as gen
from tornado.options import parse_command_line, define, options
import tornado.web as web


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
    def post(self):
        data = json_decode(self.request.body)
        response = {}
        for file in data:
            response[file] = os.access(os.path.join(
                options.root, json_file(file)), os.R_OK)
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


class TaggerHandler(web.RequestHandler):
    ImageTemplate = namedtuple(
        "ImageT", ("name", "path", "size", "format", "mode", "touched"))

    def initialize(self):
        self.logger = logging.getLogger("main")

    @staticmethod
    def discover_image(path):
        img = Image.open(path)
        return TaggerHandler.ImageTemplate(
            name=os.path.splitext(os.path.basename(path))[0],
            path=os.path.relpath(path, options.root).replace("\\", "/"),
            size=img.size, format=img.format, mode=img.mode,
            touched=os.path.exists(json_file(path)))

    def get(self):
        images = []
        app_root = os.path.abspath(os.path.dirname(__file__))
        for root, dirs, files in os.walk(options.root):
            if os.path.abspath(root).startswith(app_root):
                del dirs[:]
                continue
            for file in files:
                mime = guess_type(file)[0]
                if (mime is not None and mime.startswith("image/") and
                        mime.find("svg") < 0):
                    path = os.path.join(root, file)
                    try:
                        images.append(TaggerHandler.discover_image(path))
                    except:
                        self.logger.exception("Failed to load %s", path)
        return self.render("tagger.html",
                           images=sorted(images, key=lambda i: i.path))


define("port", default=8080, type=int, help="Port which server should listen.")
define("root", default=".", help="Root directory to scan for images.")


def main():
    parse_command_line()
    logger = logging.getLogger("main")
    logger.info("Root is set to %s", os.path.abspath(options.root))
    app = web.Application([
        ("/tagger.html", TaggerHandler),
        ("/selections", SelectionsHandler),
        ("/update", UpdateHandler),
        ("/touched", TouchedHandler),
        (r"/images/(?P<path>.*)",
         web.StaticFileHandler, {"path": options.root}),
        (r"/thumbnails/(?P<path>.*)", ThumbnailsHandler),
        (r"/((js|css|fonts|img)/.*)",
         web.StaticFileHandler,
         {'path': os.path.abspath(os.path.dirname(__file__))}),
        ("/", web.RedirectHandler, {"url": "/tagger.html", "permanent": True}),
        ("", web.RedirectHandler, {"url": "/tagger.html", "permanent": True})],
        template_path=os.path.dirname(__file__)
    )
    app.listen(options.port)
    logger.info("Listening on %d", options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    sys.exit(main())
