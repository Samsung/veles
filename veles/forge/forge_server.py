#!/usr/bin/python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 11, 2014

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

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import hashlib
import inspect
from itertools import islice
import json
import logging
import os
from smtplib import SMTPException
import uuid
from PIL import Image
import pygit2
import re
import shutil
from six import PY3
import struct
import subprocess
import sys
from tarfile import TarFile
import tempfile
from tornado import gen, web
from tornado.escape import xhtml_escape
from tornado.ioloop import IOLoop
from tornado.template import BaseLoader, Template
from tornado_smtpclient.client import SMTPAsync

from veles.cmdline import CommandLineBase
from veles.compat import from_none, PermissionError, FileNotFoundError
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
    EMAIL_REGEXP = re.compile(r"^[^@\s=]+@[^@\s=]+\.[^@\s=]+$")

    def initialize(self, server):
        self.server = server
        self.logger = logging.getLogger(
            "%s/%s" % (self.logger.name, root.common.forge.service_name))
        self.handlers = {n[len("handle_"):]: m.__get__(self) for n, m in
                         inspect.getmembers(ServiceHandler, inspect.isfunction)
                         if n.startswith("handle_")}

    @property
    def name(self):
        try:
            return self.get_argument("name")
        except web.MissingArgumentError as e:
            self.error("No 'name' argument was specified")
            raise from_none(e)

    @gen.coroutine
    def handle_details(self):
        self.finish(json_encode(self.server.details(self.name)))

    @gen.coroutine
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

    @gen.coroutine
    def handle_delete(self):
        token = self.get_argument("token")
        if self.server.scramble(token) not in self.server.tokens:
            self.error("\"%s\" is not allowed to write", token)
            self.send_error(403)
            return
        try:
            self.server.delete(token, self.name)
        except PermissionError:
            self.send_error(403)
            return
        self.finish("OK")

    @gen.coroutine
    def handle_register(self):
        email = self.get_argument("email")
        if not self.EMAIL_REGEXP.match(email):
            self.send_error(400)
            return
        token = self.server.emails.get(email)
        if token is not None:
            self.finish("You are already registered. If you forgot your token,"
                        " please unregister first.")
            return
        smtp = yield self.server.smtp()
        msg = MIMEMultipart('alternative')
        msg["Subject"] = "VelesForge registration confirmation"
        msg["From"] = sender = self.server.email_sender
        msg["To"] = email
        link = "%s://%s%s%s?query=confirm&token=%s" % (
            self.request.headers.get("HTTP_X_SCHEME", self.request.protocol),
            self.request.headers.get("Host", self.request.host),
            self.server.suburi, root.common.forge.service_name,
            self.server.generate_token(email))
        self.debug("Sending the confirmation link %s to %s", link, email)
        msg.attach(MIMEText(
            self.render_string("email.txt", link=link).decode('charmap'),
            "plain"))
        msg.attach(MIMEText(
            self.render_string("email.html", link=link).decode('charmap'),
            "html"))
        yield smtp.sendmail(sender, email, msg.as_string())
        yield smtp.quit()
        self.render_finish("confirmation.html", email=email)

    @gen.coroutine
    def handle_confirm(self):
        token = self.get_argument("token")
        if not self.server.register(token):
            self.error("Token %s was not found in the pending list", token)
            self.send_error(400)
            return
        self.render_finish("successful_registration.html", token=token)

    @gen.coroutine
    def handle_unregister(self):
        email = self.get_argument("email")
        if not self.EMAIL_REGEXP.match(email):
            self.send_error(400)
            return
        token = self.server.emails.get(email)
        if token is None:
            self.render_finish("not_registered.html")
            return
        smtp = yield self.server.smtp()
        msg = MIMEMultipart('alternative')
        msg["Subject"] = "VelesForge unregistration confirmation"
        msg["From"] = sender = self.server.email_sender
        msg["To"] = email
        link = "%s://%s%s%s?query=unconfirm&token=%s" % (
            self.request.headers.get("HTTP_X_SCHEME", self.request.protocol),
            self.request.headers.get("Host", self.request.host),
            self.server.suburi, root.common.forge.service_name, token)
        self.debug("Sending the unconfirmation link %s to %s", link, email)
        msg.attach(MIMEText(
            self.render_string("email_un.txt", link=link).decode('charmap'),
            "plain"))
        msg.attach(MIMEText(
            self.render_string("email_un.html", link=link).decode('charmap'),
            "html"))
        yield smtp.sendmail(sender, email, msg.as_string())
        yield smtp.quit()
        self.render_finish("unconfirmation.html", email=email)

    @gen.coroutine
    def handle_unconfirm(self):
        token = self.get_argument("token")
        if token not in self.server.tokens:
            self.error("Token %s was not found in the registered list", token)
            self.send_error(400)
        self.server.unregister(token)
        self.render_finish("successful_unregistration.html")

    def render_finish(self, template, **kwargs):
        self.clear()
        self.finish(self.render_string(template, **kwargs))

    @web.asynchronous
    @gen.coroutine
    def get(self):
        self.debug("GET from %s: %s", self.request.remote_ip,
                   "&".join(("%s=%s" % (k, v) for k, v in
                             self.request.arguments.items())))
        self.set_header("Content-Type", "application/json")
        yield self.handlers[self.get_argument("query")]()


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

    def _discover_version(self, rep, version):
        relative = re.match("(.+)@{(\d+)}", version)
        if relative is not None:
            version, relative = relative.groups()
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
        if relative is not None:
            treeish = next(islice(rep.walk(treeish), int(relative), None)).hex
        return treeish

    @web.asynchronous
    @gen.coroutine
    def get(self):
        name = self.get_argument("name")
        version = self.get_argument("version", "HEAD")
        self.debug("GET from %s: %s", self.request.remote_ip,
                   "&".join(self.request.arguments))
        rep = self.server.repos[name]
        treeish = self._discover_version(rep, version)
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
        self.token = self.get_argument("token", None)
        if self.server.scramble(self.token) not in self.server.tokens:
            self.error("Token \"%s\" is not registered", self.token)
            self.send_error(403)
            return
        self.metadata = None
        self.metadata_size = 0
        self.size = 0
        self.read_counter = 0
        self.cache = []

    def post(self):
        self.debug("finish POST from %s, read %d bytes",
                   self.request.remote_ip, self.read_counter)
        rep = self.server.repos.get(self.metadata["name"])
        if rep is not None and self.server.scramble(self.token) not in \
                rep.config["forge.tokens"].split():
            self.error("\"%s\" may not write to %s (owner: %s)", self.token,
                       self.metadata["name"], rep.config["forge.tokens"])
            self.send_error(403)
            return
        try:
            self.server.upload(self.token, self.metadata,
                               UploadHandler.Reader(self.cache))
            self.finish()
        except Exception as e:
            self.return_error(400, e)

    def data_received(self, chunk):
        self.read_counter += len(chunk)
        if self.metadata is not None:
            self.consume_chunk(chunk)
        else:
            if self.metadata_size == 0:
                self.metadata_size = struct.unpack('!I', chunk[:4])[0]
                if self.metadata_size > 32 * 1024:
                    self.return_error(400, "%d is a too big metadata size" %
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


class ForgeHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    def get(self):
        self.render("forge.html", items=self.server.list_full())


class ImagePageHandler(web.RequestHandler):
    def get(self):
        self.render("forge_image.html", name=self.get_argument("name"))


class InterceptingStaticFileHandler(web.StaticFileHandler):
    def initialize(self, path):
        super(InterceptingStaticFileHandler, self).initialize(path)
        self.real_root = path

    @gen.coroutine
    def get(self, path, include_body=True):
        yield super().get(self.intercepted_get(path), include_body)

    def intercepted_get(self, path):
        return path


class ThumbnailHandler(InterceptingStaticFileHandler):
    def intercepted_get(self, path):
        self.root = os.path.join(os.path.join(self.real_root, path), ".git")
        return ForgeServer.THUMBNAIL_FILE_NAME


class ImageStaticHandler(InterceptingStaticFileHandler):
    def intercepted_get(self, path):
        name, fmt = os.path.splitext(path)
        self.root = os.path.join(os.path.join(self.real_root, name), ".git")
        return ForgeServer.IMAGE_FILE_NAME % fmt[1:]


class ForgeServer(Logger):
    THUMBNAIL_FILE_NAME = "thumbnail.png"
    IMAGE_FILE_NAME = "image.%s"
    DELETED_DIR = ".deleted"

    @staticmethod
    def init_parser(sphinx=False):
        parser = ArgumentParser(
            description=CommandLineBase.LOGO if not sphinx else "",
            formatter_class=CommandLineBase.SortingRawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="Produce debug output.")
        parser.add_argument("-r", "--root", required=True,
                            help="The root directory to operate on.")
        parser.add_argument("-p", "--port", default=80, type=int,
                            help="The port to listen on.")
        parser.add_argument("--suburi", default="/",
                            help="SubURI to work behind a reverse proxy.")
        parser.add_argument("--smtp-host", required=True,
                            help="SMTP host which to use for sending emails.")
        parser.add_argument("--smtp-port", required=True,
                            help="SMTP port which to use for sending emails.")
        parser.add_argument("--smtp-user", required=True,
                            help="SMTP user which to use for sending emails.")
        parser.add_argument("--smtp-password", required=True,
                            help="SMTP password which to use for sending "
                                 "emails.")
        parser.add_argument("--email-sender", required=True,
                            help="Email \"from\" identifier.")
        return parser

    def __init__(self, args=None):
        super(ForgeServer, self).__init__()
        if args is None:
            parser = ForgeServer.init_parser()
            args = parser.parse_args()
        Logger.setup_logging(
            logging.INFO if not args.verbose else logging.DEBUG)
        for key, val in args.__dict__.items():
            setattr(self, key, val)
        self.suburi = getattr(args, "suburi", "/")
        if not self.suburi.endswith("/"):
            raise ValueError("--suburi must end with a slash (\"/\")")
        self.thread_pool = ThreadPoolExecutor(4)
        self._stop = IOLoop.instance().stop
        IOLoop.instance().stop = self.stop
        self._tokens = None
        self._tokens_timestamp = None
        self._emails = None
        self.tokens_file = getattr(self, "tokens_file",
                                   root.common.forge.tokens_file)
        self.pending_file = getattr(self, "pending_file",
                                    root.common.forge.pending_file)

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

    @property
    def tokens(self):
        timestamp = self._get_tokens_timestamp()
        if self._tokens is None or timestamp > self._tokens_timestamp:
            self._tokens = self._read_tokens(self.tokens_file)
            self._tokens_timestamp = timestamp
        return self._tokens

    @property
    def emails(self):
        timestamp = self._tokens_timestamp
        tokens = self.tokens
        if self._emails is None or self._tokens_timestamp != timestamp:
            self._emails = {e: t for t, e in tokens.items()}
        return self._emails

    def generate_token(self, email):
        pending = {e: t for t, e in
                   self._read_tokens(self.pending_file).items()}
        token = pending.get(email)
        if token is not None:
            return token
        token = str(uuid.uuid4())
        pending[email] = token
        self._append_token(self.pending_file, token, email)
        return token

    def register(self, token):
        sectok = self.scramble(token)
        if sectok in self.tokens:
            return True
        pending = self._read_tokens(self.pending_file)
        if token not in pending:
            return False
        self.tokens[sectok] = pending[token]
        del pending[token]
        self._write_tokens(self.pending_file, pending)
        self._write_tokens(self.tokens_file, self.tokens)
        self._tokens_timestamp = self._get_tokens_timestamp()
        return True

    def unregister(self, token):
        if token not in self.tokens:
            return
        del self.tokens[token]
        self._write_tokens(self.tokens_file, self.tokens)

    @staticmethod
    def scramble(value):
        sha = hashlib.sha1()
        sha.update(b"veles")
        if value:
            sha.update(value.encode('charmap'))
        return sha.hexdigest()

    def _get_tokens_timestamp(self):
        if not os.access(self.tokens_file, os.R_OK):
            with open(self.tokens_file, "w") as _:
                pass
        return os.path.getmtime(self.tokens_file)

    @staticmethod
    def _read_tokens(file):
        if not os.access(file, os.R_OK):
            with open(file, "w") as _:
                return {}
        with open(file, "r") as fin:
            tokens = {}
            for line in fin:
                if not line.strip():
                    continue
                token, email = line.strip().split("=")
                tokens[token] = email
            return tokens

    @staticmethod
    def _append_token(file, token, email):
        with open(file, "a") as fout:
            ForgeServer._write_token(fout, token, email)

    @staticmethod
    def _write_tokens(file, tokens):
        with open(file, "w") as fout:
            for pair in tokens.items():
                ForgeServer._write_token(fout, *pair)

    @staticmethod
    def _write_token(file, token, email):
        file.write("%s=%s\n" % (token, email))

    @gen.coroutine
    def smtp(self):
        smtp = SMTPAsync()
        self.debug("Accessing %s:%s...", self.smtp_host, self.smtp_port)
        yield smtp.connect(self.smtp_host, int(self.smtp_port))
        try:
            yield smtp.starttls()
        except SMTPException:
            pass
        yield smtp.login(self.smtp_user, self.smtp_password)
        return smtp

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
        self.info("Discovered %d repos", len(self.repos))

    def _generate_images(self, metadata, rep):
        where = dirname(rep.path)
        pic = metadata.get("image")
        if pic is not None:
            pic = os.path.join(where, pic)
        if pic is not None and os.access(pic, os.R_OK):
            with open(pic, "rb") as fin:
                img = Image.open(fin)
                if pic.endswith(".svg"):
                    shutil.copyfile(pic, os.path.join(
                        rep.path, ForgeServer.IMAGE_FILE_NAME % "svg"))
                full_path = os.path.join(
                    rep.path, ForgeServer.IMAGE_FILE_NAME % "webp")
                if not pic.endswith(".webp"):
                    try:
                        if img.mode not in ("RGB", "RGBA"):
                            img.convert("RGB").save(full_path, "WEBP")
                        else:
                            img.save(full_path, "WEBP")
                    except IOError:
                        self.warning(
                            "Failed to convert %s (mode %s) to WEBP format",
                            pic, pic.mode)
                else:
                    shutil.copyfile(pic, full_path)
                full_path = os.path.join(
                    rep.path, ForgeServer.IMAGE_FILE_NAME % "jpg")
                if os.path.splitext(pic)[1] not in (".jpg", ".jpeg"):
                    img.save(full_path, "JPEG")
                else:
                    shutil.copyfile(pic, full_path)
                img.thumbnail((256, 256))
                img.save(os.path.join(rep.path,
                                      ForgeServer.THUMBNAIL_FILE_NAME), "PNG")
        else:
            pic = os.path.join(rep.path, ForgeServer.IMAGE_FILE_NAME % "png")
            with tempfile.TemporaryFile() as tmp:
                retcode = subprocess.call([
                    sys.executable, "-m", "veles", "-s", "-p", "",
                    "--dry-run=init", "--workflow-graph=" + pic,
                    os.path.join(where, metadata["workflow"]),
                    os.path.join(where, metadata["configuration"])],
                    stdout=tmp.fileno(), stderr=tmp.fileno())
                tmp.seek(0)
                logs = tmp.read()
            if retcode != 0:
                self.warning("Failed to generate the thumbnail for %s",
                             metadata["name"])
                self.debug("\n" + logs.decode('charmap') + "\n")
            if os.path.exists(pic):
                img = Image.open(pic)
                img.thumbnail((256, 256))
                img.save(os.path.join(
                    rep.path, ForgeServer.THUMBNAIL_FILE_NAME), "PNG")

    def upload(self, token, metadata, reader):
        name = metadata["name"]
        version = metadata["version"]
        rep = self.repos.get(name)
        if rep is None:
            where = os.path.join(self.root, name)
            need_init = True
            if os.path.exists(where):
                self.warning("%s exists - cleared", where)
                shutil.rmtree(where)
                os.mkdir(where)
        else:
            where = dirname(rep.path)
            need_init = False
        with TarFile.open(mode="r|gz", fileobj=reader) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, where)
        if not need_init:
            self.add_version(rep, version)
        else:
            self.repos[name] = rep = pygit2.init_repository(where)
            try:
                self.add_version(rep, version)
            except Exception as e:
                shutil.rmtree(where)
                del self.repos[name]
                self.error("Failed to initialize %s", name)
                raise from_none(e)
            rep.config["forge.tokens"] = self.scramble(token)
        self._generate_images(metadata, rep)

    def add_version(self, rep, version):
        if len(rep.diff()) == 0:
            try:
                _ = rep.head
                raise Exception("No new changes")
            except pygit2.GitError:
                new = True
        else:
            new = False
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
                          [rep.head.peel().oid] if not new else [])
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
            item = [name, self._get_metadata(r)["short_description"].strip()]
            head = r.head.get_object()
            item.append(head.author.name)
            item.append(head.message.strip())
            item.append(str(datetime.utcfromtimestamp(head.commit_time)))
            yield item

    def list_full(self):
        for _, r in sorted(self.repos.items()):
            item = {}
            item.update(self._get_metadata(r))
            head = r.head.get_object()
            item["version"] = head.message.strip()
            item["updated"] = str(datetime.utcfromtimestamp(head.commit_time))
            item["versions"] = versions = []
            for i, c in enumerate(r.walk(r.head.target)):
                commit_time = str(datetime.utcfromtimestamp(c.commit_time))
                if c.message != "master" or i == 0:
                    versions.append((c.message, commit_time))
                else:
                    versions.append(("HEAD@{%d}" % i, commit_time))
            yield item

    def details(self, name):
        rep = self.repos[name]
        head = rep.head.get_object()
        return {"name": name,
                "description": self._get_metadata(rep)["long_description"],
                "version": head.message.strip(),
                "date": str(datetime.utcfromtimestamp(head.commit_time))}

    def delete(self, token, name):
        if name not in self.repos:
            raise ValueError("Package %s was not found" % name)
        rep = self.repos[name]
        try:
            if self.scramble(token) not in rep.config["forge.tokens"].split():
                self.error("\"%s\" may not write to %s (owner: %s)", token,
                           name, rep.config["forge.tokens"])
                raise PermissionError()
        except ValueError:
            raise from_none(PermissionError())
        del self.repos[name]
        path = dirname(rep.path)
        deldir = os.path.join(self.root, ForgeServer.DELETED_DIR)
        if not os.path.exists(deldir):
            os.mkdir(deldir)
        with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", prefix=name + "_", dir=deldir,
                delete=False) as fout:
            with TarFile.open(mode="w|gz", fileobj=fout) as tar:
                tar.add(path, arcname="")
            self.info("Archived %s to %s", name, fout.name)
        shutil.rmtree(path)
        self.info("Deleted %s", name)

    def uri(self, *args):
        return self.suburi + "".join(args)

    def run(self, loop=True):
        forge = root.common.forge
        self.application = web.Application([
            (self.uri(forge.service_name), ServiceHandler, {"server": self}),
            (self.uri(forge.upload_name), UploadHandler, {"server": self}),
            (self.uri(forge.fetch_name), FetchHandler, {"server": self}),
            (self.uri("forge.html"), ForgeHandler, {"server": self}),
            (self.uri("image.html"), ImagePageHandler),
            (self.uri("thumbnails/(.*)"), ThumbnailHandler,
             {"path": self.root}),
            (self.uri("images/(.*)"), ImageStaticHandler, {"path": self.root}),
            (self.uri("((js|css|fonts|img|maps)/.*)"),
             web.StaticFileHandler, {'path': root.common.web.root}),
            (self.suburi, web.RedirectHandler,
             {"url": self.uri("forge.html"), "permanent": True}),
            (self.suburi[:-1], web.RedirectHandler,
             {"url": self.uri("forge.html"), "permanent": True}),
        ], template_loader=ForgeTemplateLoader(
            root.common.web.templates, root.common.forge.email_templates))
        try:
            self.application.listen(self.port)
        except OSError as e:
            self.error("Failed to open port %d", self.port)
            raise from_none(e)
        self.info("Listening on port %d, suburi %s" % (self.port, self.suburi))
        if loop:
            IOLoop.instance().start()


class ForgeTemplateLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super(ForgeTemplateLoader, self).__init__(**kwargs)
        self.roots = [os.path.abspath(d) for d in args]

    def resolve_path(self, name, parent_path=None):
        if parent_path and not parent_path.startswith("<") and \
                not parent_path.startswith("/") and \
                not name.startswith("/"):
            for r in self.roots:
                current_path = os.path.join(r, parent_path)
                file_dir = os.path.dirname(os.path.abspath(current_path))
                relative_path = os.path.abspath(os.path.join(file_dir, name))
                if os.path.exists(relative_path) and \
                        relative_path.startswith(r):
                    name = relative_path[len(r) + 1:]
                    break
        return name

    def _create_template(self, name):
        for r in self.roots:
            path = os.path.join(r, name)
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(name)
        with open(path, "rb") as f:
            template = Template(f.read(), name=name, loader=self)
            return template


def __run__():
    return ForgeServer().run()


if __name__ == "__main__":
    __run__()
