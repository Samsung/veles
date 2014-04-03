"""
Created on Jul 12, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from copy import copy
import logging.handlers
from pymongo import MongoClient
import re
import sys


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    class ColorFormatter(logging.Formatter):
        GREEN_MARKERS = ['ok', 'finished', 'completed', 'ready', 'done',
                         'running', 'successful']
        GREEN_RE = re.compile("|".join(GREEN_MARKERS))

        def formatMessage(self, record):
            level_color = "0"
            text_color = "0"
            if record.levelno <= logging.DEBUG:
                level_color = "1;37"
                text_color = "0;37"
            elif record.levelno <= logging.INFO:
                level_color = "1;36"
                lmsg = record.message.lower()
                if Logger.ColorFormatter.GREEN_RE.search(lmsg):
                    text_color = "1;32"
            elif record.levelno <= logging.WARNING:
                level_color = "1;33"
            elif record.levelno <= logging.CRITICAL:
                level_color = "1;31"
            fmt = "\033[" + level_color + \
                "m%(levelname)s\033[0m:%(name)s:\033[" + text_color + \
                "m%(message)s\033[0m"
            return fmt % record.__dict__

    @staticmethod
    def setup(level):
        logging.basicConfig(level=level)
        if sys.stdout.isatty():
            root = logging.getLogger()
            handler = root.handlers[0]
            handler.setFormatter(Logger.ColorFormatter())

    def __init__(self, **kwargs):
        super(Logger, self).__init__()
        self._logger_ = kwargs.get("logger",
                                   logging.getLogger(self.__class__.__name__))

    @property
    def log(self):
        """Returns the logger associated with this object.
        """
        return self._logger_

    def redirect_logging_to_file(self, file_name, max_bytes=1024 * 1024,
                                 backups=9):
        handler = logging.handlers.RotatingFileHandler(
            filename=file_name, maxBytes=max_bytes, backupCount=backups
        )
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: "
                                      "%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.log.info("Redirecting output to %s", file_name)
        self.log.propagate = False
        self.log.addHandler(handler)
        self.info("Redirected output")

    @staticmethod
    def duplicate_all_logging_to_file(file_name, max_bytes=1024 * 1024):
        handler = logging.handlers.RotatingFileHandler(
            filename=file_name, maxBytes=max_bytes
        )
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: "
                                      "%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logging.info("Saving logs to %s", file_name)
        logging.getLogger().addHandler(handler)
        logging.info("Continuing")

    @staticmethod
    def duplicate_all_logging_to_mongo(addr, docid):
        handler = MongoLogHandler(addr=addr, docid=docid)
        logging.info("Saving logs to Mongo on %s", addr)
        logging.getLogger().addHandler(handler)

    def debug(self, msg, *args, **kwargs):
        self.log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log.error(msg, *args, **kwargs)

    def exception(self, msg="Exception", *args, **kwargs):
        self.log.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log.critical(msg, *args, **kwargs)


class MongoLogHandler(logging.Handler):
    def __init__(self, addr, docid, level=logging.NOTSET):
        super(MongoLogHandler, self).__init__(level)
        self._client = MongoClient("mongodb://" + addr)
        self._db = self._client.veles
        self._collection = self._db.logs
        self._id = docid

    @property
    def id(self):
        return self._id

    def emit(self, record):
        data = copy(record.__dict__)
        for bs in "levelno", "funcName", "args", "msg", "module", \
                  "processName", "msecs":
            del(data[bs])
        data["session"] = self.id
        self._collection.insert(data)
