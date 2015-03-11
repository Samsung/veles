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
from pip.backwardcompat import uses_pycache
from pip.util import normalize_path
from pip import wheel
from pkg_resources import working_set, Requirement, Distribution, \
    VersionConflict, SOURCE_DIST, PY_MAJOR
import shutil
from six.moves import input
from six.moves.urllib.parse import urlparse, urlencode
from six import PY3
from veles.import_file import try_to_import_file, is_module

if PY3:
    from importlib.util import cache_from_source
else:
    from veles.compat import cache_from_source
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
from types import ModuleType
import wget
from zope.interface import implementer

from veles import __plugins__, __name__, __version__, __root__
from veles.cmdline import CommandLineBase
from veles.compat import from_none
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
            if key not in metadata:
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
        if "image" in metadata:
            extra.append(metadata["image"])
        files = sorted({workflow, config,
                        root.common.forge.manifest}.union(extra))
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
                self.owner.debug("Metadata size is %d", len(metabytes))
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
            with TarFile.open(mode="w|gz", fileobj=body, bufsize=tbs,
                              dereference=True) as tar:
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
            if not hasattr(failure.value, "reasons"):
                try:
                    failure.raiseException()
                except:
                    self.exception("Failed to upload %s:", name)
            else:
                self.error("Failed to upload %s:\n%s", name,
                           failure.value.reasons[0].getTraceback())
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
                print("Version: %s (%s)" % (response["version"],
                                            response["date"]))
                print("Author: %s" % response["author"])
                print("")
                print(response["description"])
            except KeyError as e:
                self.exception("Response from server is not full: %s",
                               response)
                self.stop(Failure(from_none(e)), False)
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

    @action
    def delete(self):
        confirmation = "Yes, I am sure!"
        user = input("Please type \"%s\": " % confirmation)
        if user != confirmation:
            print("Refusing to delete %s because this operation was not "
                  "confirmed" % self.name)
            self.stop()
            return
        print("Deleting %s..." % self.name)
        sys.stdout.flush()

        def finished(response):
            print("Successfully deleted %s" % self.name)
            sys.stdout.flush()
            self.stop()

        def failed(failure):
            try:
                failure.raiseException()
            except:
                self.exception("Failed to delete %s:", self.name)
            self.stop(failure, False)

        url = "%s%s?query=delete&name=%s" % (
            self.base, root.common.forge.service_name, self.name)
        self.debug("Requesting %s", url)
        getPage(url.encode('charmap')).addCallbacks(callback=finished,
                                                    errback=failed)

    def _get_pkg_files(self, dist):
        """
        Shamelessly taken from pip/req.py/InstallRequirement.uninstall()
        """
        paths = set()

        def add(pth):
            paths.add(normalize_path(pth))
            if os.path.splitext(pth)[1] == '.py' and uses_pycache:
                add(cache_from_source(pth))

        pip_egg_info_path = os.path.join(
            dist.location, dist.egg_name()) + '.egg-info'
        dist_info_path = os.path.join(
            dist.location, '-'.join(dist.egg_name().split('-')[:2])
        ) + '.dist-info'
        # workaround http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=618367
        debian_egg_info_path = pip_egg_info_path.replace('-py' + PY_MAJOR, '')
        pip_egg_info_exists = os.path.exists(pip_egg_info_path)
        debian_egg_info_exists = os.path.exists(debian_egg_info_path)
        dist_info_exists = os.path.exists(dist_info_path)
        if pip_egg_info_exists or debian_egg_info_exists:
            # package installed by pip
            if pip_egg_info_exists:
                egg_info_path = pip_egg_info_path
            else:
                egg_info_path = debian_egg_info_path
            add(egg_info_path)
            if dist.has_metadata('installed-files.txt'):
                for installed_file in dist.get_metadata(
                        'installed-files.txt').splitlines():
                    path = os.path.normpath(
                        os.path.join(egg_info_path, installed_file))
                    add(path)
            elif dist.has_metadata('top_level.txt'):
                if dist.has_metadata('namespace_packages.txt'):
                    namespaces = dist.get_metadata('namespace_packages.txt')
                else:
                    namespaces = []
                for top_level_pkg in [
                    p for p in dist.get_metadata('top_level.txt').splitlines()
                        if p and p not in namespaces]:
                    path = os.path.join(dist.location, top_level_pkg)
                    add(path)
                    add(path + '.py')
                    add(path + '.pyc')
        elif dist_info_exists:
            for path in wheel.uninstallation_paths(dist):
                add(path)
        return paths

    def _scan_deps(self, module):
        print("Scanning for dependencies...")
        sys.stdout.flush()
        deps = ["veles >= " + __version__]
        for plugin in __plugins__:
            deps.append("%s >= %s" % (plugin.__name__, plugin.__version__))
        stdlib_paths = {p for p in sys.path
                        if p.find("dist-packages") < 0 and p}
        stdlib_paths.add(__root__)
        pkg_match = {}
        for dist in working_set:
            for file in self._get_pkg_files(dist):
                pkg_match[file] = dist

        for key, val in module.__dict__.items():
            if not isinstance(val, ModuleType):
                continue
            name = val.__name__
            if name in sys.builtin_module_names:
                continue
            is_stdlib = False
            for stdpath in stdlib_paths:
                if val.__file__.startswith(stdpath):
                    is_stdlib = True
                    break
            if is_stdlib:
                continue
            pkg = pkg_match.get(val.__file__)
            if pkg is not None:
                deps.append("%s >= %s" % (pkg.project_name, pkg.version))

        return deps

    @action
    def assist(self):
        metadata = {}
        base = os.path.dirname(self.path)
        modname = os.path.splitext(os.path.basename(self.path))[0]
        print("Importing %s..." % modname)
        sys.stdout.flush()
        module = try_to_import_file(self.path)
        if not is_module(module):
            self.error("Failed to import %s.\npackage: %s\ndirect: %s",
                       self.path, *module)
            self.return_code = 1
            return

        # Discover the module's dependencies
        metadata["requires"] = self._scan_deps(module)

        # Discover the workflow class
        wfcls = [None]

        def fake_load(klass, *args, **kwargs):
            wfcls[0] = klass
            return None, None

        def fake_run(*args, **kwargs):
            pass

        module.run(fake_load, fake_run)
        wfcls = wfcls[0]

        # Fill "name"
        name = wfcls.__name__.replace("Workflow", "")
        max_chars = 64
        inp = '0' * (max_chars + 1)
        while len(inp) > max_chars:
            if len(inp) > max_chars:
                print("Package name may not be longer than %d chars." %
                      max_chars)
            inp = input("Please enter the desired package name (%s): " % name)
        metadata["name"] = inp or name

        # Fill "short_description"
        max_chars = 140
        inp = "0" * (max_chars + 1)
        while len(inp) > max_chars:
            inp = input("Please enter a *short* description (<= 140 chars): ")
        metadata["short_description"] = inp

        # Fill "long_description"
        print("The long description will be taken from %s's docstring." %
              wfcls.__name__)
        metadata["long_description"] = wfcls.__doc__.strip()

        inp = input("Please introduce yourself (e.g., "
                    "\"Ivan Ivanov <i.ivanov@samsung.com>\"): ")
        metadata["author"] = inp
        metadata["workflow"] = os.path.basename(self.path)

        # Discover the configuration file
        wfn, wfext = os.path.splitext(self.path)
        wfn += "_config"
        cfgfile = wfn + wfext
        inp = cfgfile
        while (inp and not os.path.exists(inp) and
               not os.path.exists(os.path.join(base, inp))):
            inp = input("Please enter the path to the configuration file "
                        "(may be blank) (%s): " % cfgfile)
        fullcfg = inp or cfgfile
        metadata["configuration"] = os.path.basename(fullcfg)

        print("Generating %s..." % metadata["name"])
        os.mkdir(metadata["name"])
        shutil.copyfile(self.path, os.path.join(metadata["name"],
                                                metadata["workflow"]))
        destcfg = os.path.join(metadata["name"], metadata["configuration"])
        if os.path.exists(fullcfg):
            shutil.copyfile(fullcfg, destcfg)
        else:
            open(destcfg, 'w').close()
        with open(os.path.join(metadata["name"],
                               root.common.forge.manifest), "w") as fout:
            json.dump(metadata, fout, sort_keys=True, indent=4)

    action = staticmethod(action)

    @staticmethod
    def init_parser(sphinx=False):
        parser = ArgumentParser(
            description=CommandLineBase.LOGO if not sphinx else "",
            formatter_class=CommandLineBase.SortingRawDescriptionHelpFormatter)
        parser.add_argument("action", choices=ACTIONS,
                            help="Command to execute.")
        parser.add_argument(
            "-d", "--directory", default=".", dest="path",
            help="Destination directory where to save the received package;"
                 "Source package directory to upload. Path to workflow to "
                 "assist creating metadata with.")
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
            help="Force remove destination directory if it exists, "
                 "overwrite files.")
        parser.add_argument(
            "--verbose", default=False, action='store_true',
            help="Write debug messages.")
        return parser

    def __init__(self, args=None, own_reactor=True):
        super(ForgeClient, self).__init__()
        self.own_reactor = own_reactor
        if args is None:
            if sys.argv[1] == "assist":
                if len(sys.argv) < 3:
                    raise ValueError(
                        "You must specify the path to the workflow file which "
                        "you want to generate package for.")
                args = {"action": "assist", "path": sys.argv[2],
                        "verbose": False}
            else:
                parser = ForgeClient.init_parser()
                args = parser.parse_args()
        for k, v in getattr(args, "__dict__", args).items():
            setattr(self, k, v)
        try:
            Logger.setup(level=logging.DEBUG if self.verbose else logging.INFO)
        except Logger.LoggerHasBeenAlreadySetUp:
            pass
        if self.action not in ("assist",):
            self._validate_base()
            if not self.base.endswith('/'):
                self.base += '/'
        if self.action in ("details", "fetch") and not self.name:
            raise ValueError("Package name may not be empty.")
        self.action = getattr(self, self.action)

    def run(self):
        self.debug("Executing %s()", self.action.__name__)
        if self.action in (self.fetch, self.assist):
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
