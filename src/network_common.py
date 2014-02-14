"""
Created on Jan 22, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

import logging
from twisted.protocols.basic import LineReceiver


class NetworkConfigurable(object):
    """
    Stores the address and the port number.
    """

    CONFIG_ADDRESS = "address"
    CONFIG_PORT = "port"

    def __init__(self, configuration):
        """
        Parses the configuration file and loads CONFIG_ADDRESS and CONFIG_PORT
        """
        idx_semicolon = configuration.find(":")
        if idx_semicolon == -1:  # assume configuration file
            cf = open(configuration, "r")
            txt = cf.read()
            cf.close()
            self.options = eval(txt)
            if not isinstance(self.options, dict):
                raise RuntimeError("Corrupted network configuration file %s." %
                                   configuration)
            self.address = self.options[NetworkConfigurable.CONFIG_ADDRESS]
            self.port = self.options[NetworkConfigurable.CONFIG_PORT]
        else:  # assume tcp
            self.address = configuration[:idx_semicolon]
            if not self.address:
                self.address = "0.0.0.0"
            self.port = int(configuration[idx_semicolon + 1:])
        logging.info("Network configuration: %s:%d", self.address, self.port)


class StringLineReceiver(LineReceiver):
    def sendLine(self, line):
        if isinstance(line, str):
            super(StringLineReceiver, self).sendLine(line.encode())
        elif isinstance(line, bytes):
            super(StringLineReceiver, self).sendLine(line)
        else:
            raise RuntimeError("Only str and bytes are allowed.")
