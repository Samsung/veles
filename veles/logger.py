"""
Created on Jul 12, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import codecs
from copy import copy
import logging.handlers
import os
from pymongo import MongoClient
import re
import sys

from veles.external.daemon import redirect_stream
from veles.external.progressbar import ProgressBar


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    class ColorFormatter(logging.Formatter):
        GREEN_MARKERS = [' ok', 'finished', 'completed', 'ready', 'done',
                         'running', 'successful', 'saved']
        GREEN_RE = re.compile("|".join(GREEN_MARKERS))

        def formatMessage(self, record):
            level_color = "0"
            text_color = "0"
            fmt = ""
            if record.levelno <= logging.DEBUG:
                fmt = "\033[0;37m" + logging.BASIC_FORMAT + "\033[0m"
            elif record.levelno <= logging.INFO:
                level_color = "1;36"
                lmsg = record.message.lower()
                if Logger.ColorFormatter.GREEN_RE.search(lmsg):
                    text_color = "1;32"
            elif record.levelno <= logging.WARNING:
                level_color = "1;33"
            elif record.levelno <= logging.CRITICAL:
                level_color = "1;31"
            if not fmt:
                fmt = "\033[" + level_color + \
                    "m%(levelname)s\033[0m:%(name)s:\033[" + text_color + \
                    "m%(message)s\033[0m"
            return fmt % record.__dict__

        if not hasattr(logging.Formatter, "formatMessage"):
            def format(self, record):
                record.message = record.getMessage()
                if self.usesTime():
                    record.asctime = self.formatTime(record, self.datefmt)
                s = self.formatMessage(record)
                if record.exc_info:
                    if not record.exc_text:
                        record.exc_text = self.formatException(record.exc_info)
                if record.exc_text:
                    if s[-1:] != "\n":
                        s = s + "\n"
                    try:
                        s = s + record.exc_text
                    except UnicodeError:
                        s = s + \
                            record.exc_text.decode(sys.getfilesystemencoding(),
                                                   'replace')
                return s

    @staticmethod
    def setup(level):
        # Ensure UTF-8 on stdout and stderr; in some crazy environments,
        # they use 'ascii' encoding by default.
        sys.stdout, sys.stderr = (codecs.getwriter("utf-8")(s.buffer)
                                  for s in (sys.stdout, sys.stderr))
        sys.stdout.encoding = sys.stderr.encoding = "utf-8"
        # Set basic log level
        logging.basicConfig(level=level, stream=sys.stdout)
        ProgressBar().logger.level = level
        # Turn on colors in case of an interactive tty
        if sys.stdout.isatty():
            root = logging.getLogger()
            handler = root.handlers[0]
            handler.setFormatter(Logger.ColorFormatter())

    def __init__(self, **kwargs):
        super(Logger, self).__init__()
        self._logger_ = kwargs.get("logger",
                                   logging.getLogger(self.__class__.__name__))

    @property
    def logger(self):
        """Returns the logger associated with this object.
        """
        return self._logger_

    @staticmethod
    def redirect_all_logging_to_file(file_name, max_bytes=1024 * 1024,
                                     backups=1):
        handler = logging.handlers.RotatingFileHandler(
            filename=file_name, maxBytes=max_bytes, backupCount=backups,
            encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: "
                                      "%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logging.info("Saving logs to %s", file_name)
        if not sys.stdout.isatty():
            logging.getLogger().handlers[0] = handler
            sys.stderr.flush()
            stderr = open("%s.stderr%s" % os.path.splitext(file_name), 'a',
                          encoding="utf-8")
            redirect_stream(sys.stderr, stderr)
            sys.stderr = stderr
        logging.getLogger().addFilter(handler)
        logging.info("Continuing to log in %s", file_name)

    @staticmethod
    def duplicate_all_logging_to_mongo(addr, docid):
        handler = MongoLogHandler(addr=addr, docid=docid)
        logging.info("Saving logs to Mongo on %s", addr)
        logging.getLogger().addHandler(handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg="Exception", *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


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
            del data[bs]
        data["session"] = self.id
        self._collection.insert(data)
