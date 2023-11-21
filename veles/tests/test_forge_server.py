# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 12, 2014

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


from __future__ import print_function
from collections import namedtuple
import logging
import os
import shutil
import socket
import struct
import sys
from tarfile import TarFile
import threading
from time import time
import unittest

from numpy.random import randint
import pygit2
from six import BytesIO
from tornado import gen
from tornado.httpclient import HTTPClient, HTTPRequest, HTTPError
from tornado.ioloop import IOLoop

from veles import __root__
from veles.config import root
from veles.forge.forge_server import ForgeServer


while True:
    PORT = 8067 + randint(-1000, 1000)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if probe.connect_ex(('127.0.0.1', PORT)):
        probe.close()
        break

ForgeServerArgs = namedtuple(
    "ForgeServerArgs", ("root", "port", "smtp_host", "smtp_port", "smtp_user",
                        "smtp_password", "email_sender", "verbose"))


class FakeSMTP(object):
    @gen.coroutine
    def sendmail(self, sender, addr, msg):
        self.sender = sender
        self.addr = addr
        self.msg = msg

    @gen.coroutine
    def quit(self):
        pass


class TestForgeServer(unittest.TestCase):
    ioloop_thread = None
    server = None
    fake_smtp = FakeSMTP()

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
                    
                
                safe_extract(tar, os.path.join(base,name))
        sys.stderr = sys.stdout
        cls.server = ForgeServer(ForgeServerArgs(
            base, PORT, "smtp_host", 25, "user", "password", "from", True))
        cls.server.smtp = TestForgeServer.smtp.__get__(cls.server)
        cls.server.run(loop=False)
        cls.ioloop_thread.start()

    def setUp(self):
        self.client = HTTPClient()
        self.base = os.path.join(__root__, "veles/tests/forge")

    @classmethod
    def tearDownClass(cls):
        IOLoop.instance().stop()
        cls.ioloop_thread.join()

    @gen.coroutine
    def smtp(self):
        return TestForgeServer.fake_smtp

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
        TestForgeServer.server.tokens[
            TestForgeServer.server.scramble("secret")] = "user@domain.com"
        try:
            try:
                self.client.fetch(HTTPRequest(
                    method='POST',
                    url="http://localhost:%d/upload?token=n" % PORT,
                    body=self._compose_upload("second_bad.tar.gz")))
                self.fail("HTTPError was not thrown")
            except HTTPError as e:
                self.assertEqual(e.response.code, 403)
            try:
                self.client.fetch(HTTPRequest(
                    method='POST', url="http://localhost:%d/upload?token=%s" %
                                       (PORT, "secret"),
                    body=self._compose_upload("second_bad.tar.gz")))
                self.fail("HTTPError was not thrown")
            except HTTPError as e:
                self.assertGreaterEqual(
                    e.response.body.find(b'No new changes'), 0)
            response = self.client.fetch(HTTPRequest(
                method='POST', url="http://localhost:%d/upload?token=%s" % (
                    PORT, "secret"),
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
        TestForgeServer.server.tokens[
            TestForgeServer.server.scramble("secret")] = "user@domain.com"
        try:
            response = self.client.fetch(HTTPRequest(
                method='POST', url="http://localhost:%d/upload?token=%s" % (
                    PORT, "secret"),
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
        TestForgeServer.server.tokens[
            TestForgeServer.server.scramble("secret")] = "user@domain.com"
        try:
            response = self.client.fetch(
                "http://localhost:%d/service?query=delete&name=Second&"
                "token=secret" % PORT)
            self.assertEqual(response.body, b'OK')
            self.assertFalse(os.path.exists(dirname))
            self.assertTrue(os.path.exists(deldir))
            backup_file = list(os.walk(deldir))[0][2][0]
            with TarFile.open(os.path.join(deldir, backup_file)) as tar:
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
                    
                
                safe_extract(tar, deldir)
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

    def test_register_bad(self):
        self.assertRaises(
            HTTPError, self.client.fetch,
            "http://localhost:%d/service?query=register&"
            "email=bademail.com" % PORT)
        self.assertRaises(
            HTTPError, self.client.fetch,
            "http://localhost:%d/service?query=register&"
            "email=bad=a@email.com" % PORT)
        self.assertRaises(
            HTTPError, self.client.fetch,
            "http://localhost:%d/service?query=register&"
            "email=" % PORT)

    def do_register(self):
        TestForgeServer.server._tokens = {}
        TestForgeServer.server._tokens_timestamp = time()
        TestForgeServer.server._emails = {}
        response = self.client.fetch(
            "http://localhost:%d/service?query=register&"
            "email=user@domain.com" % PORT)
        self.assertTrue(
            b"The confirmation email has been sent to user@domain.com"
            in response.body)
        pos = self.fake_smtp.msg.find("token=") + len("token=")
        token = self.fake_smtp.msg[pos:pos+36]
        response = self.client.fetch(
            "http://localhost:%d/service?query=confirm&"
            "token=%s" % (PORT, token))
        self.assertTrue(
            ("Registered, your token is %s" % token).encode("utf-8")
            in response.body)
        TestForgeServer.server._emails = {
            "user@domain.com":  self.server.scramble(token)}
        return token

    def test_register_unregister(self):
        token = self.do_register()
        response = self.client.fetch(
            "http://localhost:%d/service?query=unregister&"
            "email=user@domain.com" % PORT)
        self.assertTrue(
            b"The confirmation email has been sent to user@domain.com"
            in response.body)
        response = self.client.fetch(
            "http://localhost:%d/service?query=unconfirm&"
            "token=%s" % (PORT, self.server.scramble(token)))
        self.assertTrue(b"Successfully unregistered" in response.body)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
