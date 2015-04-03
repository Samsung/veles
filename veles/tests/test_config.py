#!/usr/bin/python3 -O
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mart 31, 2014

Unit test for global config

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


import logging
from six import StringIO
import unittest

from veles.config import root, Config, get, validate_kwargs
from veles.pickle2 import pickle


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.was_warning = False

    def test_temp_config(self):
        logging.info("Will test automatic appearance of an"
                     " intermediate Config object")
        test = Config("")
        test.test_.test_value = 5
        self.assertEqual(test.test_.test_value, 5,
                         "Not equal values on second level")
        logging.info("All ok")

    def test_update_multi_line(self):
        logging.info("Will test multi-line update function")
        test = Config("")

        test.test_value = 0.01
        test.test_.test_value = 5
        test.test_.test_str = "Test"

        test.update({"test_value": 0.5,
                     "test_": {"test_value": 12,
                               "test_str": "new test"}})
        self.assertEqual(test.test_value, 0.5,
                         "Not equal values on first level")
        self.assertEqual(test.test_.test_value, 12,
                         "Not equal values on second level")
        self.assertEqual(test.test_.test_str, "new test", "Not equal strings")

        logging.info("All Ok")

    def test_update_one_line(self):
        logging.info("Will test one-line update function")
        test = Config("")

        test.update({"test_value": 0.5,
                     "test_": {"test_value": 12,
                               "test_str": "new test"}})

        test.test_value = 0.01
        test.test_.test_value = 5
        test.test_.test_str = "Test"

        self.assertEqual(test.test_value, 0.01,
                         "Not equal values on first level")
        self.assertEqual(test.test_.test_value, 5,
                         "Not equal values on second level")
        self.assertEqual(test.test_.test_str, "Test", "Not equal strings")

        logging.info("All Ok")

    def test_get_config(self):
        logging.info("Will test get function")
        test = Config("")
        test.test_value = 0.01
        test.test_value = get(test.test_value, 0.5)
        self.assertEqual(test.test_value, 0.01,
                         "No right value in get")
        test.test_value2 = get(test.test_value2, 0.5)
        self.assertEqual(test.test_value2, 0.5,
                         "No right defolt value in get")

    def warning(self, *args, **kwargs):
        self.was_warning = True

    def test_print(self):
        f = StringIO()
        root.print_(file=f)
        value = f.getvalue()
        self.assertGreater(len(value), 0)
        self.assertTrue("'common'" in value)
        self.assertTrue("'engine'" in value)

    def test_validate(self):
        validate_kwargs(self, first=root.unit_test, second=root.nonexistnt)
        self.assertTrue(self.was_warning)

    def test_pickling(self):
        test = Config("root")
        test.a = 10
        test.b = 20
        test.update({"c": {"inner": "string"}})
        self.assertIsInstance(test.c, Config)
        buffer = pickle.dumps(test)
        new_test = pickle.loads(buffer)
        self.assertEqual(new_test.__path__, "root")
        self.assertEqual(new_test.a, 10)
        self.assertEqual(new_test.b, 20)
        self.assertIsInstance(new_test.c, Config)
        self.assertEqual(new_test.c.__path__, "root.c")
        self.assertEqual(new_test.c.inner, "string")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Will test config unit")
    unittest.main()
