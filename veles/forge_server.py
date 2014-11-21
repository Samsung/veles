#!/usr/bin/python3
"""
Created on Nov 11, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from argparse import ArgumentParser
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
import os
import re
import shutil
import pygit2
from six import PY3
import struct
from tarfile import TarFile
from tornado import gen, web
from tornado.escape import xhtml_escape
from tornado.ioloop import IOLoop

from veles.cmdline import CommandLineBase
from veles.config import root
from veles.forge_common import REQUIRED_MANIFEST_FIELDS, validate_requires
from veles.logger import Logger


def dirname(path):
    return os.path.abspath(os.path.join(path, ".."))


def json_encode(value):
    return json.dumps(value, sort_keys=True).replace("</", "<\\/")


class HandlerBase(web.RequestHandler, Logger):
    def write_error(self, status_code, **kwargs):
        if hasattr(self, "error_message"):
            if isinstance(self.error_message, BaseException):
                import traceback
                self.write("<html><body><pre>%s</pre></body></html>" %
                           xhtml_escape(''.join(traceback.format_exc())))
            else:
                self.write("<html><body>Error: %s</body></html>" %
                           xhtml_escape(self.error_message))
        else:
            super(HandlerBase, self).write_error(status_code, **kwargs)

    def return_error(self, code, message):
        if isinstance(message, BaseException):
            self.exception("Returning %d (see the exception below)", code)
        else:
            self.error("Returning %d with \"%s\"", code, message)
        self.error_message = message
        self.send_error(code)


class ServiceHandler(HandlerBase):
    def initialize(self, server):
        self.server = server
        self.logger = logging.getLogger(
            "%s/%s" % (self.logger.name, root.common.forge.service_name))
        self.handlers = {"details": self.handle_details,
                         "list": self.handle_list}

    def handle_details(self):
        try:
            name = self.get_argument("name")
        except web.MissingArgumentError:
            self.error("No 'name' argument was specified")
            raise
        self.finish(json_encode(self.server.details(name)))

    def handle_list(self):
        self.write('[')
        first = True
        for item in self.server.list():
            if not first:
                self.write(',')
            else:
                first = False
            self.write(json_encode(item))
        self.finish(']')

    @web.asynchronous
    def get(self):
        self.debug("GET from %s: %s", self.request.remote_ip,
                   "&".join(("%s=%s" % (k, v) for k, v in
                             self.request.arguments.items())))
        self.set_header("Content-Type", "application/json")
        self.handlers[self.get_argument("query")]()


class FetchHandler(HandlerBase):
    class Writer(object):
        def __init__(self, handler):
            self.handler = handler

        def write(self, data):
            IOLoop.instance().add_callback(self.handler.write, data)

    def initialize(self, server):
        self.server = server
        self.logger = logging.getLogger(
            "%s/%s" % (self.logger.name, root.common.forge.fetch_name))

    @web.asynchronous
    @gen.coroutine
    def get(self):
        name = self.get_argument("name")
        version = self.get_argument("version", "HEAD")
        self.debug("GET from %s: %s", self.request.remote_ip,
                   "&".join(self.request.arguments))
        rep = self.server.repos[name]
        try:
            treeish = rep.lookup_branch(version).target
        except:
            try:
                treeish = rep.lookup_reference("refs/tags/" + version).target
            except:
                try:
                    treeish = rep.lookup_reference(version).target
                except:
                    self.return_error(
                        404, "Reference %s was not found" % version)
        if isinstance(treeish, pygit2.Oid):
            treeish = treeish.hex
        if treeish.startswith("refs/"):
            treeish = rep.lookup_reference(treeish).target
        self.debug("Resolved %s to %s", version, treeish)
        self.set_header("Content-Type", "application/x-gzip")
        self.set_header("Content-Disposition",
                        "attachment; filename=%s.tar.gz" % name)
        with TarFile.open(mode="w|gz",
                          fileobj=FetchHandler.Writer(self)) as tar:
            yield self.server.thread_pool.submit(
                rep.write_archive, treeish, tar)

        def finish():
            self.finish()
            self.debug("Finished")

        IOLoop.instance().add_callback(finish)


@web.stream_request_body
class UploadHandler(HandlerBase):
    class Reader(object):
        def __init__(self, chunks):
            self.chunks = chunks
            self.pos = [0, 0]

        def read(self, size):
            used = []
            cur_size = 0
            while cur_size < size and self.pos[0] < len(self.chunks):
                if PY3:
                    mv = memoryview(self.chunks[self.pos[0]])
                else:
                    mv = self.chunks[self.pos[0]]
                delta = cur_size + len(mv) - self.pos[1] - size
                if delta > 0:
                    end = self.pos[1] + size - cur_size
                    self.pos[1] = end
                else:
                    end = len(mv)
                    self.pos[0] += 1
                    self.pos[1] = 0
                part = mv[self.pos[1]:end]
                cur_size += len(part)
                used.append(part)
            return b''.join(used)

    def initialize(self, server):
        self.server = server
        self.logger = logging.getLogger(
            "%s/%s" % (self.logger.name, root.common.forge.upload_name))

    def prepare(self):
        self.debug("start POST from %s", self.request.remote_ip)
        self.metadata = None
        self.size = 0
        self.read_counter = 0
        self.cache = []

    def post(self):
        self.debug("finish POST from %s, read %d bytes",
                   self.request.remote_ip, self.read_counter)
        try:
            self.server.upload(self.metadata, UploadHandler.Reader(self.cache))
            self.finish()
        except Exception as e:
            self.return_error(400, e)

    def data_received(self, chunk):
        self.read_counter += len(chunk)
        if self.metadata is not None:
            self.consume_chunk(chunk)
        else:
            if self.size == 0:
                self.metadata_size = struct.unpack('!I', chunk[:4])[0]
                if self.metadata_size > 32 * 1024:
                    self.return_error(400, "%d is too big metadata size" %
                                           self.metadata_size)
                    return
                self.debug("metadata size is %d", self.metadata_size)
                if PY3:
                    chunk = memoryview(chunk)[4:]
                else:
                    chunk = chunk[4:]
            if self.size + len(chunk) > self.metadata_size:
                self.debug("approached a breakpoint")
                rem = self.metadata_size - self.size
                if PY3:
                    self.cache.append(memoryview(chunk)[:rem])
                else:
                    self.cache.append(chunk[:rem])
                try:
                    self.read_metadata()
                except Exception as e:
                    self.return_error(400, e)
                    return
                self.size = rem
                if PY3:
                    self.cache = [memoryview(chunk)[rem:]]
                else:
                    self.cache = [chunk[rem:]]
            else:
                self.consume_chunk(chunk)

    def consume_chunk(self, chunk):
        self.size += len(chunk)
        self.cache.append(chunk)

    def read_metadata(self):
        try:
            self.metadata = json.loads(b''.join(self.cache).decode("UTF-8"))
        except:
            raise ValueError("Failed to load metadata JSON")
        if not isinstance(self.metadata, dict):
            raise ValueError("Wrong format of metadata")
        for key in REQUIRED_MANIFEST_FIELDS.union({"version"}):
            if key not in self.metadata:
                raise ValueError("%s not found in metadata" % key)
        validate_requires(self.metadata['requires'])


ForgeServerArgs = namedtuple("ForgeServerArgs", ("root", "port"))


class ForgeServer(Logger):
    @staticmethod
    def init_parser_(sphinx=False):
        parser = ArgumentParser(
            description=CommandLineBase.LOGO if not sphinx else "")
        parser.add_argument("-r", "--root", help="The root directory to "
                                                 "operate on.")
        parser.add_argument("-p", "--port", default=80, type=int,
                            help="The port to listen on.")
        return parser

    def __init__(self, args=None):
        super(ForgeServer, self).__init__()
        if args is None:
            parser = ForgeServer.init_parser_()
            args = parser.parse_args()
        self.root = args.root
        self.port = args.port
        self.thread_pool = ThreadPoolExecutor(4)
        self._stop = IOLoop.instance().stop
        IOLoop.instance().stop = self.stop

    def stop(self):
        self._stop()
        self.thread_pool.shutdown()

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        if not isinstance(value, int):
            raise TypeError("port must be an instance of type int")
        if value < 1:
            raise ValueError("value must be > 0")
        self._port = value

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        if not os.path.exists(value):
            raise ValueError("%s does not exist" % value)
        self._root = value
        self._build_db()

    @property
    def repos(self):
        return self._repos

    def _get_metadata(self, rep):
        with open(os.path.join(dirname(rep.path), root.common.forge.manifest),
                  "rb") as fin:
            return json.loads(fin.read().decode('UTF-8'))

    def _build_db(self):
        self._repos = {}
        for r in (pygit2.Repository(d) for d in (
                os.path.join(self.root, d) for d in os.listdir(self.root))
                if os.path.exists(os.path.join(d, ".git"))):
            try:
                self.repos[self._get_metadata(r)["name"]] = r
            except Exception as e:
                self.warning("Repository %s looks corrupted and was skipped "
                             "(%s)", dirname(r.path), e)

    def upload(self, metadata, reader):
        name = metadata["name"]
        version = metadata["version"]
        rep = self.repos.get(name)
        if rep is None:
            where = os.path.join(self.root, name)
            need_init = True
            if os.path.exists(where):
                self.warning("%s exists - cleared")
                shutil.rmtree(where)
                os.mkdir(where)
        else:
            where = dirname(rep.path)
            need_init = False
        with TarFile.open(mode="r|gz", fileobj=reader) as tar:
            tar.extractall(where)
        if not need_init:
            self.add_version(rep, version)
        else:
            self.repos[name] = rep = pygit2.init_repository(where)
            self.add_version(rep, version)

    def add_version(self, rep, version):
        if len(rep.diff()) == 0:
            raise Exception("No changes")
        metadata = self._get_metadata(rep)
        rep.index.read()
        base = dirname(rep.path)
        for path, dirs, files in os.walk(dirname(rep.path)):
            if os.path.basename(path) == ".git":
                del dirs[:]
                continue
            for file in files:
                rep.index.add(os.path.relpath(os.path.join(path, file), base))
        author = metadata["author"].strip()
        email = re.search(r"(?<=<)[^@>]+@[^@>]+(?=>$)", author)
        if email is not None:
            email = email.group()
            author = author[:-(len(email) + 2)].strip()
        rep.create_commit('HEAD', pygit2.Signature(author, email or ""),
                          pygit2.Signature("VelesForge", ""),
                          version, rep.index.write_tree(),
                          [rep.head.peel().oid])
        rep.index.write()
        if version != "master":
            try:
                rep.lookup_reference("refs/tags/" + version)
                raise Exception("Tag %s already exists for %s" % (
                    version, metadata["name"]))
            except:
                rep.create_tag(
                    version, rep.head.peel().oid, pygit2.GIT_OBJ_COMMIT,
                    pygit2.Signature("VelesForge", ""), "")

    def list(self):
        for name, r in sorted(self.repos.items()):
            item = [name, self._get_metadata(r)["short_description"]]
            head = r.head.get_object()
            item.append(head.author.name)
            item.append(head.message.strip())
            item.append(str(datetime.utcfromtimestamp(head.commit_time)))
            yield item

    def details(self, name):
        rep = self.repos[name]
        head = rep.head.get_object()
        return {"name": name,
                "description": self._get_metadata(rep)["long_description"],
                "version": head.message.strip(),
                "date": str(datetime.utcfromtimestamp(head.commit_time))}

    def run(self, loop=True):
        forge = root.common.forge
        self.application = web.Application([
            ("/" + forge.service_name, ServiceHandler, {"server": self}),
            ("/" + forge.upload_name, UploadHandler, {"server": self}),
            ("/" + forge.fetch_name, FetchHandler, {"server": self}),
        ])
        self.application.listen(self.port)
        self.info("Listening on port %d" % self.port)
        if loop:
            IOLoop.instance().start()


def __run__():
    return ForgeServer().run()

if __name__ == "__main__":
    __run__()
