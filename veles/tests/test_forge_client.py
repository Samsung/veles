"""
Created on Nov 10, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from __future__ import print_function
import io
import json
from numpy.random import randint
import os
import shutil
from six import StringIO
import struct
import sys
import tarfile
import tempfile
import threading
from twisted.internet import reactor
from twisted.web.server import Site
from twisted.web.resource import Resource
import unittest
import wget

from veles import __root__, __plugins__, __version__
from veles.config import root
from veles.forge_client import ForgeClient, ForgeClientArgs


PORT = 8068 + randint(-1000, 1000)


class Router(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.page_class = None
        self.callback = lambda name, request: None

    def getChild(self, name, request):
        self.callback(name, request)
        return self.page_class()


class TestForgeClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = Router()
        reactor.listenTCP(PORT, Site(cls.router))
        cls.thread = threading.Thread(target=reactor.run, args=(False,))
        cls.thread.start()
        download = wget.download

        def patched_download(*args, **kwargs):
            if isinstance(sys.stdout, StringIO):
                sys.stdout.fileno = lambda: 0
            download(*args, **kwargs)

        wget.download = download

    @classmethod
    def tearDownClass(cls):
        reactor.callFromThread(reactor.stop)
        cls.thread.join()

    def setUp(self):
        self.sync = threading.Event()
        self.sync.success = None

    def tearDown(self):
        TestForgeClient.router.page_class = None
        TestForgeClient.router.callback = lambda name, request: None

    def succeed(self, _=None):
        self.sync.success = True
        self.sync.set()

    def fail(self, msg=None):
        if msg is not None:
            super().fail(msg)
        else:
            self.sync.success = False
            self.sync.set()

    def sync(fn):
        def wrapped_sync(self, *args, **kwargs):
            stdout = sys.stdout
            fn(self, *args, **kwargs)
            d = self.client.run()
            if "_bad" in fn.__name__:
                d.addCallback(self.fail).addErrback(self.succeed)
            elif "_good" in fn.__name__:
                d.addCallback(self.succeed).addErrback(self.fail)
            else:
                raise TypeError("@sync may not be applied to %s()",
                                fn.__name__)
            self.sync.wait()
            self.assertTrue(self.sync.success)
            after = getattr(self, "after" + fn.__name__.replace("test", ""),
                            None)
            if after is not None:
                print("Continuing in %s()" % after.__name__,
                      file=stdout)
                after()

        wrapped_sync.__name__ = fn.__name__
        return wrapped_sync

    @sync
    def test_list_bad(self):
        class BadServicePage(Resource):
            def render_GET(self, request):
                return b"..."

        def check_name(name, request):
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = BadServicePage
        TestForgeClient.router.callback = check_name
        self.client = ForgeClient(
            ForgeClientArgs("list", "http://localhost:%d" % PORT, True),
            False)

    @sync
    def test_list_good(self):
        class ListPage(Resource):
            def render_GET(self, request):
                return json.dumps([(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]).encode()

        def check_name(name, request):
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = ListPage
        TestForgeClient.router.callback = check_name
        self.client = ForgeClient(
            ForgeClientArgs("list", "http://localhost:%d" % PORT, True),
            False)
        self.stdout = sys.stdout
        try:
            fno = sys.stdout.fileno()
        except (io.UnsupportedOperation, AttributeError):
            fno = 0
        sys.stdout = StringIO()
        sys.stdout.fileno = lambda: fno

    def after_list_good(self):
        out = sys.stdout.getvalue()
        sys.stdout = self.stdout
        self.assertEqual(out,
                         """+------+-------------+--------+---------+------+
| Name | Description | Author | Version | Date |
+------+-------------+--------+---------+------+
|  1   |      2      |   3    |    4    |  5   |
|  6   |      7      |   8    |    9    |  10  |
+------+-------------+--------+---------+------+
""")

    @sync
    def test_details_bad(self):
        class BadServicePage(Resource):
            def render_GET(self, request):
                return b"..."

        def check_name(name, request):
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = BadServicePage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, True)
        args.name = "First"
        self.client = ForgeClient(args, False)

    @sync
    def test_details_bad_dict(self):
        class DetailsPage(Resource):
            def render_GET(self, request):
                return json.dumps({"name": "test"}).encode()

        def check_name(name, request):
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = DetailsPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, True)
        args.name = "First"
        self.client = ForgeClient(args, False)

    @sync
    def test_details_good(self):
        class DetailsPage(Resource):
            def render_GET(self, request):
                return json.dumps({
                    "name": "Test Model",
                    "description": "Really long text telling about how this "
                                   "model is awesome.",
                    "version": "1.0.0",
                    "author": "VELES team",
                    "date": "<date here>"}).encode()

        def check_name(name, request):
            self.assertEqual(request.args[b"name"][0], b"First")
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = DetailsPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, True)
        args.name = "First"
        self.client = ForgeClient(args, False)
        self.stdout = sys.stdout
        sys.stdout = StringIO()

    def after_details_good(self):
        out = sys.stdout.getvalue()
        sys.stdout = self.stdout
        self.assertEqual(out,
                         """Test Model
==========

Version: 1.0.0 (<date here>)
Author: VELES team

Really long text telling about how this model is awesome.
""")

    def test_fetch_bad(self):
        class BadFetchPage(Resource):
            def render_GET(self, request):
                return b"..."

        def check_name(name, request):
            self.assertEqual(request.args[b"name"][0], b"First")
            self.assertEqual(request.args[b"version"][0], b"master")
            self.assertEqual(b'fetch', name)

        TestForgeClient.router.page_class = BadFetchPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("fetch", "http://localhost:%d" % PORT, True)
        args.name = "First"
        args.version = "master"
        args.force = True
        td = tempfile.mkdtemp()
        args.path = td
        try:
            self.client = ForgeClient(args, False)
            self.assertRaises(tarfile.ReadError, self.client.run)
        finally:
            self.sync.set()
            shutil.rmtree(td)

    def test_fetch_good(self):
        class FetchPage(Resource):
            def render_GET(self, request):
                with open(os.path.join(__root__,
                                       "veles/tests/forge/second_good.tar.gz"),
                          "rb") as fin:
                    return fin.read()

        def check_name(name, request):
            self.assertEqual(request.args[b"name"][0], b"First")
            self.assertEqual(request.args[b"version"][0], b"master")
            self.assertEqual(b'fetch', name)

        TestForgeClient.router.page_class = FetchPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("fetch", "http://localhost:%d" % PORT, True)
        args.name = "First"
        args.version = "master"
        args.force = True
        td = tempfile.mkdtemp()
        args.path = td
        try:
            self.client = ForgeClient(args, False)
            path, md = self.client.run()
            self.assertEqual(path, td)
            self.assertEqual(md["name"], "Second")
            self.assertEqual(md["workflow"], "second.py")
            self.assertEqual({'config.py', 'manifest.json', 'second.py'},
                             set(os.listdir(td)))
        finally:
            self.sync.set()
            shutil.rmtree(td)

    @sync
    def test_upload_bad(self):
        class BadUploadPage(Resource):
            def render_POST(self, request):
                return b"!"

        def check_name(name, request):
            self.assertEqual(b'upload', name)

        TestForgeClient.router.page_class = BadUploadPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("upload", "http://localhost:%d" % PORT, True)
        args.path = os.path.join(__root__, "veles/tests")
        args.version = "2.4"
        self.client = ForgeClient(args, False)

    @sync
    def test_upload_good(self):
        this = self
        manifest_size = os.path.getsize(
            os.path.join(__root__, "veles/tests/forge/First/%s" %
                         root.common.forge.manifest)) + 1
        # "+ 1" is present because of the UTF-8 signature, see
        # metabytes = json.dumps(metadata, sort_keys = True).encode('UTF-8')
        # inside ForgeClient.upload.ForgeBodyProducer.startProducing

        class UploadPage(Resource):
            def render_POST(self, request):
                input = request.content.read()
                this.assertEqual(input[:4], struct.pack('!I', manifest_size))
                this.assertEqual(
                    input[4:manifest_size+4],
                    ('{"author": "VELES Team", "configuration": '
                     '"workflow_config.py", "long_description": "First test '
                     'model LOOONG text", "name": "First", '
                     '"requires": ["veles>=%s"], '
                     '"short_description": "First test model", '
                     '"version": "2.4", "workflow": "workflow.py"}'
                     % __version__).encode())
                this.assertEqual(len(input) - manifest_size - 4, 314)
                return b"!"

        def check_name(name, request):
            self.assertEqual(b'upload', name)

        TestForgeClient.router.page_class = UploadPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("upload", "http://localhost:%d" % PORT, True)
        args.path = os.path.join(__root__, "veles/tests/forge/First")
        args.version = "2.4"
        self.client = ForgeClient(args, False)

    def test_fetch_bad_deps(self):
        buf = StringIO()
        stderr = sys.stderr
        sys.stderr = buf
        self.client = ForgeClient(
            ForgeClientArgs("list", "http://localhost:%d" % PORT, True),
            False)
        self.client._check_deps({"requires": ["abrakadabra>=0.1.2"]})
        sys.stderr = stderr
        self.assertGreaterEqual(
            buf.getvalue().find("Unsatisfied package requirements"), 0)
        self.assertGreaterEqual(
            buf.getvalue().find("abrakadabra>=0.1.2"), 0)
        import numpy
        __plugins__.add(numpy)
        numpy.__version__ = "1.0"
        buf = StringIO()
        sys.stderr = buf
        self.client._check_deps({"requires": [
            "numpy>=1000.1", "twisted>=14.0"]})
        sys.stderr = stderr
        self.assertGreaterEqual(
            buf.getvalue().find("Unsatisfied VELES requirements"), 0)
        self.assertGreaterEqual(
            buf.getvalue().find("numpy>=1000.1"), 0)

    sync = staticmethod(sync)

if __name__ == "__main__":
    unittest.main()
