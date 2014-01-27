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

    def __init__(self, config_file):
        """
        Parses the configuration file and loads CONFIG_ADDRESS and CONFIG_PORT
        """
        cf = open(config_file, "r")
        txt = cf.read()
        cf.close()
        options = eval(txt)
        if not isinstance(options, dict):
            raise RuntimeError("Corrupted network configuration file %s." %
                               config_file)
        self.address = options[NetworkConfigurable.CONFIG_ADDRESS]
        self.port = options[NetworkConfigurable.CONFIG_PORT]
        logging.info("Network configuration:    %s:%d",
                     self.address, self.port)


class StringLineReceiver(LineReceiver):
    def sendLine(self, line):
        if isinstance(line, str):
            super(StringLineReceiver, self).sendLine(line.encode())
        elif isinstance(line, bytes):
            super(StringLineReceiver, self).sendLine(line)
        else:
            raise RuntimeError("Only str and bytes are allowed.")
