"""
Created on Jan 22, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import six
from twisted.protocols.basic import LineReceiver
import uuid

from veles.logger import Logger


class NetworkAgent(Logger):
    """
    Stores the address and the port number.
    """

    CONFIG_ADDRESS = "address"
    CONFIG_PORT = "port"

    def __init__(self, configuration, workflow):
        """
        Parses the configuration file and loads CONFIG_ADDRESS and CONFIG_PORT
        """
        super(NetworkAgent, self).__init__()
        self._mid = None
        self._pid = None
        idx_semicolon = configuration.find(":")
        assert idx_semicolon >= 0
        self.address = configuration[:idx_semicolon]
        if not self.address:
            self.address = "0.0.0.0"
        self.port = int(configuration[idx_semicolon + 1:])
        self.debug("Network configuration: %s:%d", self.address, self.port)
        self._workflow = workflow
        self._launcher = workflow.workflow

    @property
    def pid(self):
        if self._pid is None:
            self._pid = os.getpid()
        return self._pid

    @property
    def mid(self):
        if self._mid is None:
            with open("/var/lib/dbus/machine-id") as midfd:
                self._mid = "%s-%x" % (midfd.read()[:-1], uuid.getnode())
        return self._mid

    @property
    def workflow(self):
        return self._workflow

    @property
    def launcher(self):
        return self._launcher


class StringLineReceiver(LineReceiver, object):
    def sendLine(self, line):
        if isinstance(line, str):
            if six.PY3:
                super(StringLineReceiver, self).sendLine(line.encode())
            else:
                LineReceiver.sendLine(self, line.encode())
        elif isinstance(line, bytes):
            if six.PY3:
                super(StringLineReceiver, self).sendLine(line)
            else:
                LineReceiver.sendLine(self, line)
        else:
            raise RuntimeError("Only str and bytes are allowed.")


class IDLogger(Logger):
    def __init__(self, logger=None, log_id=None):
        super(IDLogger, self).__init__(logger=logger)
        self.id = log_id

    def change_log_message(self, msg):
        return "%s: %s" % (self.id or "<none>", msg)
