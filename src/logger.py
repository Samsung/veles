"""
Created on Jul 12, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""
import logging


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    def __init__(self):
        self.init_unpickled()

    def init_unpickled(self):
        self.logger_ = logging.getLogger(self.__class__.__name__)

    def log(self):
        """Returns the logger associated with this object.
        """
        return self.logger_
