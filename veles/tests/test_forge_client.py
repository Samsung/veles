"""
Created on Nov 10, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from __future__ import print_function
import json
import logging
import os
import shutil
from six import StringIO
import sys
import tarfile
import tempfile
import threading
from twisted.internet import reactor
from twisted.web.server import Site
from twisted.web.resource import Resource
import unittest

from veles import __root__
from veles.forge_client import ForgeClient, ForgeClientArgs


PORT = 8068


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
        self.sync.success = False
        self.sync.set()

    def sync(fn):
        def wrapped_sync(self, *args, **kwargs):
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
                      file=sys.stderr)
                after()

        wrapped_sync.__name__ = fn.__name__ + '_sync'
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
            ForgeClientArgs("list", "http://localhost:%d" % PORT, False),
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
            ForgeClientArgs("list", "http://localhost:%d" % PORT, False),
            False)
        self.stdout = sys.stdout
        sys.stdout = StringIO()

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
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, False)
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
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, False)
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
                    "date": "<date here>"}).encode()

        def check_name(name, request):
            self.assertEqual(request.args[b"name"][0], b"First")
            self.assertEqual(b'service', name)

        TestForgeClient.router.page_class = DetailsPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("details", "http://localhost:%d" % PORT, False)
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
        args = ForgeClientArgs("fetch", "http://localhost:%d" % PORT, False)
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
        args = ForgeClientArgs("fetch", "http://localhost:%d" % PORT, False)
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
        args = ForgeClientArgs("upload", "http://localhost:%d" % PORT, False)
        args.path = os.path.join(__root__, "veles/tests")
        args.version = "2.4"
        self.client = ForgeClient(args, False)

    @sync
    def test_upload_good(self):
        this = self

        class UploadPage(Resource):
            def render_POST(self, request):
                input = request.content.read()
                this.assertEqual(input[:4], b'\x00\x00\x00\xda')
                this.assertEqual(
                    input[4:218+4],
                    b'{"author": "VELES Team", "configuration": '
                    b'"workflow_config.py", "long_description": "First test '
                    b'model LOOONG text", "name": "First", '
                    b'"short_description": "First test model", '
                    b'"version": "2.4", "workflow": "workflow.py"}')
                this.assertEqual(len(input) - 218 - 4, 175)
                return b"!"

        def check_name(name, request):
            self.assertEqual(b'upload', name)

        TestForgeClient.router.page_class = UploadPage
        TestForgeClient.router.callback = check_name
        args = ForgeClientArgs("upload", "http://localhost:%d" % PORT, False)
        args.path = os.path.join(__root__, "veles/tests/forge/First")
        args.version = "2.4"
        self.client = ForgeClient(args, False)

    sync = staticmethod(sync)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
