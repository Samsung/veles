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
            path = os.path.join(base, os.path.join(name, ".git"))
            if os.path.exists(path):
                shutil.rmtree(path)
            with TarFile.open(
                    os.path.join(base, "%s.git.tar.gz" % name)) as tar:
                tar.extractall(os.path.join(base, name))
        cls.server = ForgeServer(ForgeServerArgs(base, PORT))
        cls.server.run(loop=False)
        cls.ioloop_thread.start()

    def setUp(self):
        self.client = HTTPClient()
        self.base = os.path.join(__root__, "veles/tests/forge")

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
        name = os.path.splitext(os.path.splitext(file)[0])[0]
        mfn = os.path.join(self.base, name + "_" + root.common.forge.manifest)
        tarfn = os.path.join(self.base, file)
        body = bytearray(4 + os.path.getsize(mfn) + os.path.getsize(tarfn))
        body[:4] = struct.pack('!I', os.path.getsize(mfn))
        with open(mfn, 'rb') as fin:
            body[4:4 + os.path.getsize(mfn)] = fin.read()
        with open(tarfn, 'rb') as fin:
            body[4 + os.path.getsize(mfn):] = fin.read()
        logging.debug("Will send %d (incl. metadata: 4 + %d) bytes",
                      len(body), os.path.getsize(mfn))
        return bytes(body)

    def test_upload(self):
        src_path = os.path.join(self.base, "Second")
        bak_path = os.path.join(self.base, "Second.bak")
        shutil.copytree(src_path, bak_path)
        try:
            try:
                self.client.fetch(HTTPRequest(
                    method='POST', url="http://localhost:%d/upload" % PORT,
                    body=self._compose_upload("second_bad.tar.gz")))
                self.fail("HTTPError was not thrown")
            except HTTPError as e:
                self.assertGreaterEqual(
                    e.response.body.find(b'No new changes'), 0)
            response = self.client.fetch(HTTPRequest(
                method='POST', url="http://localhost:%d/upload" % PORT,
                body=self._compose_upload("second_good.tar.gz")))
            self.assertEqual(response.reason, 'OK')
            rep = pygit2.Repository(os.path.join(self.base, "Second"))
            self.assertEqual("2.0.0", rep.head.get_object().message)
            self.assertEqual(
                2, len([c for c in rep.walk(rep.head.target,
                                            pygit2.GIT_SORT_TOPOLOGICAL)]))
            self.assertIsNotNone(rep.lookup_reference("refs/tags/2.0.0"))
        finally:
            shutil.rmtree(src_path)
            shutil.move(bak_path, src_path)

    def test_upload_new(self):
        try:
            response = self.client.fetch(HTTPRequest(
                method='POST', url="http://localhost:%d/upload" % PORT,
                body=self._compose_upload("First2.tar.gz")))
            self.assertEqual(response.reason, 'OK')
            rep = pygit2.Repository(os.path.join(self.base, "First2"))
            self.assertEqual("master", rep.head.get_object().message)
            self.assertEqual(
                1, len([c for c in rep.walk(
                    rep.head.target, pygit2.GIT_SORT_TOPOLOGICAL)]))
        finally:
            rpath = os.path.join(self.base, "First2")
            if os.path.exists(rpath):
                shutil.rmtree(rpath)

    def test_delete(self):
        dirname = os.path.join(self.base, "Second")
        shutil.copytree(dirname, dirname + ".bak")
        deldir = os.path.join(self.base, ForgeServer.DELETED_DIR)
        try:
            response = self.client.fetch(
                "http://localhost:%d/service?query=delete&name=Second" % PORT)
            self.assertEqual(response.body, b'OK')
            self.assertFalse(os.path.exists(dirname))
            self.assertTrue(os.path.exists(deldir))
            backup_file = list(os.walk(deldir))[0][2][0]
            with TarFile.open(os.path.join(deldir, backup_file)) as tar:
                tar.extractall(deldir)
            orig_files = set(list(os.walk(dirname + ".bak"))[0][2])
            extr_files = set(list(os.walk(deldir))[0][2])
            self.assertEqual(extr_files.difference(orig_files), {backup_file})
        finally:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            shutil.copytree(dirname + ".bak", dirname)
            shutil.rmtree(dirname + ".bak")
            if os.path.exists(deldir):
                shutil.rmtree(deldir)
            TestForgeServer.server._build_db()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
