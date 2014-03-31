#!/usr/bin/python3.3 -O
"""
Created on Mart 31, 2014

Unit test for global config

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""


import logging
import unittest

from veles.config import root, Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True

    def test_temp_config(self):
        logging.info("Will test automatic appearance of an"
                     " intermediate Config object")
        test = Config()
        test.test_.test_value = 5
        self.assertEqual(test.test_.test_value, 5,
                         "Not equal values on second level")
        logging.info("All ok")

    def test_update_multi_line(self):
        logging.info("Will test multi-line update function")
        test = Config()

        test.test_value = 0.01
        test.test_.test_value = 5
        test.test_.test_str = "Test"

        test.update = {"test_value": 0.5,
                       "test_": {"test_value": 12,
                                 "test_str": "new test"}
                       }
        self.assertEqual(test.test_value, 0.5,
                         "Not equal values on first level")
        self.assertEqual(test.test_.test_value, 12,
                         "Not equal values on second level")
        self.assertEqual(test.test_.test_str, "new test", "Not equal strings")

        logging.info("All Ok")

    def test_update_one_line(self):
        logging.info("Will test one-line update function")
        test = Config()

        test.update = {"test_value": 0.5,
                       "test_": {"test_value": 12,
                                 "test_str": "new test"}
                       }

        test.test_value = 0.01
        test.test_.test_value = 5
        test.test_.test_str = "Test"

        self.assertEqual(test.test_value, 0.01,
                         "Not equal values on first level")
        self.assertEqual(test.test_.test_value, 5,
                         "Not equal values on second level")
        self.assertEqual(test.test_.test_str, "Test", "Not equal strings")

        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Will test config unit")
    unittest.main()
