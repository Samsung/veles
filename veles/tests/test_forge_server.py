"""
Created on Nov 12, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
from numpy.random import randint
import os
import pygit2
import shutil
from six import BytesIO
import struct
from tarfile import TarFile
import threading
from tornado.httpclient import HTTPClient, HTTPRequest, HTTPError
from tornado.ioloop import IOLoop
import unittest

from veles import __root__
from veles.config import root
from veles.forge_server import ForgeServer, ForgeServerArgs


PORT = 8067 + randint(-1000, 1000)


class TestForgeServer(unittest.TestCase):
    ioloop_thread = None
    server = None

    @classmethod
    def setUpClass(cls):
        cls.ioloop_thread = threading.Thread(
            target=IOLoop.instance().start)
        base = os.path.join(__root__, "veles/tests/forge")
        for name in ("First", "Second"):
            shutil.rmtree(os.path.join(base, os.path.join(name, ".git")))
            with TarFile.open(
                    os.path.join(base, "%s.git.tar.gz" % name)) as tar:
                tar.extractall(os.path.join(base, name))
        cls.server = ForgeServer(ForgeServerArgs(base, PORT))
        cls.server.run(loop=False)
        cls.ioloop_thread.start()

    def setUp(self):
        self.client = HTTPClient()

    @classmethod
    def tearDownClass(cls):
        IOLoop.instance().stop()
        cls.ioloop_thread.join()

    def test_list(self):
        response = self.client.fetch(
            "http://localhost:%d/service?query=list" % PORT)
        self.assertEqual(
            response.body,
            b'[["First", "First test model", "Vadim Markovtsev", "master",'
            b' "2014-11-17 11:02:24"],["Second", "Second test model", '
            b'"Vadim Markovtsev", "1.0.0", "2014-11-17 11:00:46"]]')

    def test_details(self):
        response = self.client.fetch(
            "http://localhost:%d/service?query=details&name=Second" % PORT)
        self.assertEqual(
            response.body,
            b'{"date": "2014-11-17 11:00:46", "description": "Second test '
            b'model LOOONG textz", "name": "Second", "version": "1.0.0"}')

    def _test_fetch_case(self, query, files):
        response = self.client.fetch(
            "http://localhost:%d/fetch?%s" % (PORT, query))
        with TarFile.open(fileobj=BytesIO(response.body)) as tar:
            self.assertEqual(set(tar.getnames()), files)

    def test_fetch(self):
        files = {'manifest.json', 'workflow.py', 'workflow_config.py'}
        self._test_fetch_case("name=First", files)
        self._test_fetch_case("name=First&version=master", files)
        files = {'manifest.json', 'second.py', 'config.py'}
        self._test_fetch_case("name=Second&version=1.0.0", files)

    def _compose_upload(self, file):
        base = os.path.join(__root__, "veles/tests/forge")
        mfn = os.path.join(base, root.common.forge.manifest)
        tarfn = os.path.join(base, file)
        body = bytearray(4 + os.path.getsize(mfn) + os.path.getsize(tarfn))
        body[:4] = struct.pack('!I', os.path.getsize(mfn))
        with open(mfn, 'rb') as fin:
            body[4:4 + os.path.getsize(mfn)] = fin.read()
        with open(tarfn, 'rb') as fin:
            body[4 + os.path.getsize(mfn):] = fin.read()
        return bytes(body)

    def test_upload(self):
        base = os.path.join(__root__, "veles/tests/forge")
        src_path = os.path.join(base, "Second")
        bak_path = os.path.join(base, "Second.bak")
        shutil.copytree(src_path, bak_path)
        try:
            try:
                self.client.fetch(HTTPRequest(
                    method='POST', url="http://localhost:%d/upload" % PORT,
                    body=self._compose_upload("second_bad.tar.gz")))
                self.fail("HTTPError was not thrown")
            except HTTPError as e:
                self.assertGreaterEqual(e.response.body.find(b'No changes'), 0)
            response = self.client.fetch(HTTPRequest(
                method='POST', url="http://localhost:%d/upload" % PORT,
                body=self._compose_upload("second_good.tar.gz")))
            self.assertEqual(response.reason, 'OK')
            rep = pygit2.Repository(os.path.join(base, "Second"))
            self.assertEqual("2.0.0", rep.head.get_object().message)
            self.assertEqual(
                2, len([c for c in rep.walk(rep.head.target,
                                            pygit2.GIT_SORT_TOPOLOGICAL)]))
            self.assertIsNotNone(rep.lookup_reference("refs/tags/2.0.0"))
        finally:
            shutil.rmtree(src_path)
            shutil.move(bak_path, src_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
