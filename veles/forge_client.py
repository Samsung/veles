#!/usr/bin/python3
"""
Created on Nov 10, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from __future__ import print_function
from argparse import ArgumentParser
import json
import logging
import os
from pkg_resources import working_set, Requirement, Distribution, \
    VersionConflict, SOURCE_DIST
import shutil
from six.moves.urllib.parse import urlparse, urlencode
import struct
import sys
from tarfile import TarFile, TarInfo
import threading
from twisted.internet import reactor, task
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure
from twisted.web.client import Agent, getPage
from twisted.web.iweb import IBodyProducer, UNKNOWN_LENGTH
from twisted.web.http_headers import Headers
import wget
from zope.interface import implementer

from veles import __plugins__, __name__, __version__
from veles.cmdline import CommandLineBase
from veles.config import root
from veles.external.prettytable import PrettyTable
from veles.forge_common import REQUIRED_MANIFEST_FIELDS, validate_requires
from veles.logger import Logger


ACTIONS = set()


class ForgeClientArgs(object):
    def __init__(self, action, base, verbose=False):
        self.action = action
        self.base = base
        self.verbose = verbose


class ForgeClient(Logger):
    UPLOAD_TAR_BUFFER_SIZE = 1024 * 1024
    UPLOAD_PENDING_BUFFERS = 4

    def action(fn):
        ACTIONS.add(fn.__name__)
        return fn

    @action
    def fetch(self):
        self.path = self.path or ""
        if not self.path or (os.path.exists(self.path) and
                             os.path.samefile(self.path, os.getcwd())):
            self.path = os.path.join(self.path, "%s:%s" % (self.name,
                                                           self.version))
        if os.path.exists(self.path):
            if self.force:
                shutil.rmtree(self.path)
            else:
                raise ValueError("Directory %s already exists and -f/--force "
                                 "option was not specified" % self.path)
        os.mkdir(self.path)
        url = "%s%s?%s" % (
            self.base, root.common.forge.fetch_name,
            urlencode((("name", self.name), ("version", self.version))))
        self.debug("Requesting %s", url)
        fn = wget.download(url, self.path)
        print("")
        self.debug("Downloaded %s", fn)
        with TarFile.open(fn) as tar:
            tar.extractall(self.path)
        os.remove(fn)
        self.info("Put %s to %s", os.listdir(self.path), self.path)
        metadata = self._parse_metadata(self.path)
        self._check_deps(metadata)
        return self.path, metadata

    @action
    def upload(self):
        try:
            metadata = self._parse_metadata(self.path)
        except Exception as e:
            self.exception("Failed to upload %s:", self.path)
            self.stop(Failure(e), False)
            return

        for key in REQUIRED_MANIFEST_FIELDS:
            if not key in metadata:
                raise ValueError("No \"%s\" in %s" %
                                 (key, root.common.forge.manifest))
        requires = metadata["requires"]
        validate_requires(requires)
        vreqfound = False
        for req in requires:
            if Requirement.parse(req).project_name == __name__:
                vreqfound = True
                break
        if not vreqfound:
            velreq = __name__ + ">=" + __version__
            self.warning("No VELES core requirement was specified. "
                         "Appended %s", velreq)
            requires.append(velreq)
        metadata["version"] = self.version
        name = metadata["name"]
        workflow = metadata["workflow"]
        config = metadata["configuration"]
        extra = metadata.get("files", [])
        files = [workflow, config] + extra
        self.info("Uploading %s...", name)
        agent = Agent(reactor)
        headers = Headers({b'User-Agent': [b'twisted']})

        # We will send the following:
        # 4 bytes with length of metadata in JSON format
        # metadata in JSON format
        # tar.gz package

        @implementer(IBodyProducer)
        class ForgeBodyProducer(object):
            def __init__(self, owner):
                self.owner = owner
                self.writer = None
                self.length = UNKNOWN_LENGTH
                self.finished = Deferred()
                self.consumer = None
                self.memory = 0
                self.event = threading.Event()

            def startProducing(self, consumer):
                metabytes = json.dumps(
                    metadata, sort_keys=True).encode('UTF-8')
                consumer.write(struct.pack("!I", len(metabytes)))
                consumer.write(metabytes)
                self.consumer = consumer
                self.writer.start()
                return self.finished

            def pauseProducing(self):
                pass

            def resumeProducing(self):
                pass

            def stopProducing(self):
                if self.writer.is_alive():
                    self.writer.join()

            def send(self, data):
                self.consumer.write(data)
                self.memory -= len(data)
                self.event.set()

            def write(self, data):
                self.memory += len(data)
                while self.memory > ForgeClient.UPLOAD_PENDING_BUFFERS * \
                        ForgeClient.UPLOAD_TAR_BUFFER_SIZE:
                    self.owner.debug("Suspended tar pipeline")
                    self.event.wait()
                    self.event.clear()

                self.owner.debug("Scheduling send(%d bytes), pending %d bytes",
                                 len(data), self.memory - len(data))
                reactor.callFromThread(self.send, data)

            def close(self):
                self.owner.debug("Closing, %d bytes are pending", self.memory)
                reactor.callFromThread(self.finished.callback, None)

        body = ForgeBodyProducer(self)

        def write_package():
            tbs = ForgeClient.UPLOAD_TAR_BUFFER_SIZE
            with TarFile.open(mode="w|gz", fileobj=body, bufsize=tbs) as tar:
                for file in files:
                    self.debug("Sending %s", file)
                    ti = TarInfo(file)
                    fp = os.path.join(self.path, file)
                    ti.size = os.path.getsize(fp)
                    ti.mode = 0o666
                    with open(fp, "rb") as fd:
                        tar.addfile(ti, fileobj=fd)
            body.close()

        writer = threading.Thread(target=write_package)
        body.writer = writer

        def finished(response):
            self.debug("Response from server: %s", response.code)
            if response.code != 200:
                message = "Response code is %d" % response.code
                self.error(message)
                self.stop(Failure(Exception(message)), False)
            else:
                self.stop()

        def failed(failure):
            try:
                failure.raiseException()
            except:
                self.exception("Failed to upload %s:", name)
            self.stop(failure, False)

        url = self.base + root.common.forge.upload_name
        self.debug("Sending the request to %s", url)
        d = agent.request(
            b'POST', url.encode('charmap'), headers=headers, bodyProducer=body)
        d.addCallback(finished)
        d.addErrback(failed)

    @action
    def list(self):
        def finished(response):
            self.debug("Received %s", response)
            try:
                response = json.loads(response.decode('UTF-8'))
            except ValueError as e:
                self.exception("Failed to parse the response from server: %s",
                               response)
                self.stop(Failure(e), False)
                return
            table = PrettyTable("Name", "Description", "Author", "Version",
                                "Date")
            for item in response:
                table.add_row(*item)
            print(table)
            sys.stdout.flush()
            self.stop()

        def failed(failure):
            self.return_code = 1
            try:
                failure.raiseException()
            except:
                self.exception("Failed to list the available models:")
            self.stop(failure, False)

        url = self.base + root.common.forge.service_name + "?query=list"
        self.debug("Requesting %s", url)
        getPage(url.encode('charmap')).addCallbacks(callback=finished,
                                                    errback=failed)

    @action
    def details(self):
        def finished(response):
            try:
                response = json.loads(response.decode('UTF-8'))
            except ValueError as e:
                self.exception("Failed to parse the response from server: %s",
                               response)
                self.stop(Failure(e), False)
                return
            try:
                print(response["name"])
                print("=" * len(response["name"]))
                print("")
                print("")
                print("Version: %s (%s)" % (response["version"],
                                            response["date"]))
                print("")
                print(response["description"])
            except KeyError as e:
                self.exception("Response from server is not full: %s",
                               response)
                self.stop(Failure(e), False)
                return
            sys.stdout.flush()
            self.stop()

        def failed(failure):
            try:
                failure.raiseException()
            except:
                self.exception("Failed to retrieve details for %s:", self.name)
            self.stop(failure, False)

        url = "%s%s?query=details&name=%s" % (
            self.base, root.common.forge.service_name, self.name)
        self.debug("Requesting %s", url)
        getPage(url.encode('charmap')).addCallbacks(callback=finished,
                                                    errback=failed)

    action = staticmethod(action)

    @staticmethod
    def init_parser_(sphinx=False):
        parser = ArgumentParser(
            description=CommandLineBase.LOGO if not sphinx else "")
        parser.add_argument("action", choices=ACTIONS,
                            help="Command to execute.")
        parser.add_argument(
            "-d", "--directory", default=".", dest="path",
            help="Destination directory where to save the received package;"
                 "Source package directory to upload.")
        parser.add_argument(
            "-n", "--name", default="",
            help="Package name to download/show details about.")
        parser.add_argument(
            "-v", "--version", default="master",
            help="Uploaded/downloaded package version.")
        parser.add_argument(
            "-s", "--server", dest="base", required=True,
            help="Address of VelesForge server, e.g., http://host:8080/forge")
        parser.add_argument(
            "-f", "--force", default=False, action='store_true',
            help="Force remove destination directory if it exists.")
        parser.add_argument(
            "--verbose", default=False, action='store_true',
            help="Write debug messages.")
        return parser

    def __init__(self, args=None, own_reactor=True):
        super(ForgeClient, self).__init__()
        self.own_reactor = own_reactor
        if args is None:
            parser = ForgeClient.init_parser_()
            args = parser.parse_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self._validate_base()
        if not self.base.endswith('/'):
            self.base += '/'
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO)
        if self.action in ("details", "fetch") and not self.name:
            raise ValueError("Package name may not be empty.")
        self.action = getattr(self, self.action)

    def run(self):
        self.debug("Executing %s()", self.action.__name__)
        if self.action == self.fetch:
            # Much simplier
            return self.action()
        d = task.deferLater(reactor, 0, self.action)
        d.addErrback(self.stop)
        if self.own_reactor:
            self.return_code = 0
            reactor.callWhenRunning(self.debug, "Reactor is running")
            reactor.run()
            return self.return_code
        else:
            self.run_deferred = Deferred()
            return self.run_deferred

    def stop(self, failure=None, throw=True):
        if self.own_reactor:
            reactor.stop()
        else:
            if failure is not None:
                self.run_deferred.errback(failure)
            else:
                self.run_deferred.callback(None)
        if failure is not None and throw:
            failure.raiseException()

    def _check_deps(self, metadata):
        plugins = {k.__name__.replace('_', '-'): Distribution(
            os.path.dirname(k.__file__), None, k.__name__, k.__version__,
            None, SOURCE_DIST) for k in __plugins__}
        failed_general = set()
        failed_veles = set()
        for rreq in metadata["requires"]:
            req = Requirement.parse(rreq)
            if req.project_name in plugins:
                if plugins[req.project_name] not in req:
                    failed_veles.add(req)
            else:
                try:
                    working_set.find(req).project_name
                except (AttributeError, VersionConflict):
                    failed_general.add(req)
        if len(failed_general) > 0:
            print("Unsatisfied package requirements:", file=sys.stderr)
            print(", ".join((str(f) for f in failed_general)), file=sys.stderr)
        if len(failed_veles):
            print("Unsatisfied VELES requirements:", file=sys.stderr)
            print(", ".join((str(f) for f in failed_veles)), file=sys.stderr)

    def _validate_base(self):
        pres = urlparse(self.base)
        if pres.scheme not in ("http", "https") or not pres.netloc or \
                pres.params or pres.query or pres.fragment:
            raise ValueError("Invalid URL: %s" % self.base)

    def _parse_metadata(self, path):
        # default encoding may be ascii, open in safe mode
        with open(os.path.join(path, root.common.forge.manifest), 'rb') as fin:
            return json.loads(fin.read().decode("utf-8"))


def __run__(own_reactor=True):
    return ForgeClient(own_reactor=own_reactor).run()


if __name__ == "__main__":
    __run__()
