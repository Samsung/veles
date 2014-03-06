"""
Created on Jan 22, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

import logging
import os
from twisted.protocols.basic import LineReceiver
import uuid


class NetworkAgent(object):
    """
    Stores the address and the port number.
    """

    CONFIG_ADDRESS = "address"
    CONFIG_PORT = "port"

    def __init__(self, configuration):
        """
        Parses the configuration file and loads CONFIG_ADDRESS and CONFIG_PORT
        """
        super(NetworkAgent, self).__init__()
        self._mid = None
        self._pid = None
        idx_semicolon = configuration.find(":")
        if idx_semicolon == -1:  # assume configuration file
            cf = open(configuration, "r")
            txt = cf.read()
            cf.close()
            self.options = eval(txt)
            if not isinstance(self.options, dict):
                raise RuntimeError("Corrupted network configuration file %s." %
                                   configuration)
            self.address = self.options[NetworkAgent.CONFIG_ADDRESS]
            self.port = self.options[NetworkAgent.CONFIG_PORT]
        else:  # assume tcp
            self.address = configuration[:idx_semicolon]
            if not self.address:
                self.address = "0.0.0.0"
            self.port = int(configuration[idx_semicolon + 1:])
        logging.info("Network configuration: %s:%d", self.address, self.port)

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


class StringLineReceiver(LineReceiver):
    def sendLine(self, line):
        if isinstance(line, str):
            super(StringLineReceiver, self).sendLine(line.encode())
        elif isinstance(line, bytes):
            super(StringLineReceiver, self).sendLine(line)
        else:
            raise RuntimeError("Only str and bytes are allowed.")
