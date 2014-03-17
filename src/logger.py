"""
Created on Jul 12, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import logging.handlers


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    def __init__(self, logger=None):
        if logger is None:
            self.init_unpickled()
        else:
            self.logger_ = logger

    def init_unpickled(self):
        self.logger_ = logging.getLogger(self.__class__.__name__)

    @property
    def log(self):
        """Returns the logger associated with this object.
        """
        return self.logger_

    def redirect_logging_to_file(self, file_name, max_bytes=1024 * 1024,
                                 backups=9):
        handler = logging.handlers.RotatingFileHandler(
            filename=file_name, maxBytes=max_bytes, backupCount=backups
        )
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: "
                                      "%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger_.info("Redirecting output to %s", file_name)
        self.logger_.addHandler(handler)

    def debug(self, msg, *args, **kwargs):
        self.logger_.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger_.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger_.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger_.error(msg, *args, **kwargs)

    def exception(self, msg="Exception", *args, **kwargs):
        self.logger_.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger_.critical(msg, *args, **kwargs)
