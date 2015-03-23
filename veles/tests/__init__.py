"""
Created on April 10, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.dummy import DummyWorkflow, DummyLauncher
from veles.tests.timeout import timeout
from veles.tests.accelerated_test import AcceleratedTest, multi_device, \
    assign_backend
from veles.config import root
root.common.disable_plotting = True
