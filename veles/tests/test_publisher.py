#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on April 24, 2015

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
import os
import platform
from six import string_types, PY3
from time import time
import unittest
from zope.interface import implementer

from veles.config import Config
from veles.dummy import DummyWorkflow
from veles.launcher import Launcher
from veles.logger import Logger
from veles.loader import Loader, ILoader
from veles.pickle2 import pickle, best_protocol
from veles.publishing import Publisher
from veles.publishing.confluence_backend import ConfluenceBackend


@implementer(ILoader)
class DummyLoader(Loader):
    def load_data(self):
        pass

    def create_minibatch_data(self):
        pass

    def fill_minibatch(self):
        pass


class TestPublisher(unittest.TestCase, Logger):
    def __init__(self, methodName="runTest"):
        Logger.__init__(self)
        unittest.TestCase.__init__(self, methodName)

    def setUp(self):
        self.workflow = wf = DummyWorkflow()
        self.loader = loader = DummyLoader(wf, normalization_type="mean_disp")
        loader.link_from(wf.start_point)
        self.publisher = Publisher(wf, backends={})
        self.publisher.link_from(loader)
        wf.end_point.link_from(self.publisher)
        loader._has_labels = True
        loader.class_lengths[0] = 0
        loader.class_lengths[1] = 1000
        loader.class_lengths[2] = 10000
        loader._labels_mapping = {"comedy": 0, "drama": 1, "action": 2,
                                  "hard porn": 3}
        loader.test_diff_labels = {"comedy": 0, "drama": 0, "action": 0,
                                   "hard porn": 0}
        loader.valid_diff_labels = {"comedy": 240, "drama": 260, "action": 235,
                                    "hard porn": 265}
        loader.train_diff_labels = {"comedy": 2433, "drama": 2567,
                                    "action": 2355, "hard porn": 2645}
        loader.epoch_number = 30
        self.publisher.loader_unit = loader
        if PY3:
            Launcher._generate_workflow_graphs(self)
        else:
            Launcher._generate_workflow_graphs.__func__(self)
        self.publisher.initialize()

    def test_init_info(self):
        info = self.publisher.gather_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["plots"], {})
        self.assertIsInstance(info["workflow_graph"], dict)
        for fmt in "png", "svg":
            self.assertIsInstance(info["workflow_graph"][fmt], bytes)
        self.assertEqual(info["name"], self.workflow.name)
        self.assertEqual(info["description"], self.workflow.__doc__)
        self.assertEqual(info["id"], self.workflow.workflow.id)
        self.assertEqual(info["python"],
                         "%s %s" % (platform.python_implementation(),
                                    platform.python_version()))
        self.assertEqual(info["pid"], os.getpid())
        self.assertEqual(info["logid"], self.workflow.workflow.log_id)
        self.assertIsInstance(info["config_root"], Config)
        self.assertIsInstance(info["config_text"], str)
        mins, secs = divmod(time() - self.workflow.workflow.start_time, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        self.assertEqual(info["days"], days)
        self.assertEqual(info["hours"], hours)
        self.assertEqual(info["mins"], mins)
        self.assertEqual(info["labels"], tuple(self.loader.labels_mapping))
        self.assertEqual(
            info["label_stats"], (self.loader.test_diff_labels,
                                  self.loader.valid_diff_labels,
                                  self.loader.train_diff_labels))
        self.assertEqual(info["class_lengths"],
                         tuple(self.loader.class_lengths))
        self.assertEqual(info["total_samples"], self.loader.total_samples)
        self.assertEqual(info["epochs"], self.loader.epoch_number)
        self.assertEqual(info["normalization"], self.loader.normalization_type)
        self.assertEqual(info["normalization_parameters"],
                         self.loader.normalization_parameters)
        self.assertIsInstance(info["unit_run_times_by_class"], dict)
        self.assertIsInstance(info["unit_run_times_by_name"], dict)
        self.assertIsInstance(info["seeds"], list)

    def test_confluence_render(self):
        info = self.publisher.gather_info()
        info["errors_pt"] = 0, 0.5, 0.6
        info["seeds"].append(b"abcd1234")
        conf = ConfluenceBackend(None, server="", username="", password="",
                                 space="")
        content = super(ConfluenceBackend, conf).render(info)
        self.assertIsInstance(content, string_types)
        self.info("Confluence backend rendered:\n%s", content)

    def test_pickle(self):
        wf = DummyWorkflow()
        publisher = Publisher(wf, backends={})
        loader = DummyLoader(wf, normalization_type="mean_disp")
        publisher.link_from(loader)
        wf.end_point.link_from(publisher)
        publisher.loader_unit = loader
        publisher.initialize()
        publ = pickle.dumps(publisher, protocol=best_protocol)
        publisher = pickle.loads(publ)

    @property
    def is_slave(self):
        return False

    @property
    def reports_web_status(self):
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
