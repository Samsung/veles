"""
Created on Jan 22, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

import logging


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
        options = eval(open(config_file, "r").read())
        self.address = options[NetworkConfigurable.CONFIG_ADDRESS]
        self.port = options[NetworkConfigurable.CONFIG_PORT]
        logging.info("Network configuration:    %s:%d",
                     self.address, self.port)
