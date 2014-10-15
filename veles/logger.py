"""
Created on Jul 12, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import bson
import codecs
from copy import copy
import logging.handlers
import os
from pymongo import MongoClient
import re
from six import PY3
import sys
import time

from veles.config import __path__
from veles.error import Bug
from veles.external.daemon import redirect_stream
from veles.external.progressbar import ProgressBar


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    SET_UP = False

    class LoggerHasBeenAlreadySetUp(Exception):
        pass

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
        if Logger.SET_UP:
            raise Logger.LoggerHasBeenAlreadySetUp()
        Logger.SET_UP = True
        Logger.ensure_utf8_streams()
        sys.stdout.encoding = sys.stderr.encoding = "utf-8"
        # Set basic log level
        logging.basicConfig(level=level, stream=sys.stdout)
        ProgressBar().logger.level = level
        # Turn on colors in case of an interactive out tty
        if sys.stdout.isatty():
            root = logging.getLogger()
            handler = root.handlers[0]
            handler.setFormatter(Logger.ColorFormatter())

    @staticmethod
    def ensure_utf8_streams():
        """Forces UTF-8 on stdout and stderr; in some crazy environments,
        they use 'ascii' encoding by default
        """
        if PY3:
            sys.stdout, sys.stderr = (codecs.getwriter("utf-8")(s.buffer)
                                      for s in (sys.stdout, sys.stderr))
        else:
            sys.stdout, sys.stderr = (codecs.getwriter("utf-8")(s)
                                      for s in (sys.stdout, sys.stderr))

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
        if not sys.stdin.isatty():
            logging.getLogger().handlers[0] = handler
            sys.stderr.flush()
            stderr = open("%s.stderr%s" % os.path.splitext(file_name), 'a',
                          encoding="utf-8")
            redirect_stream(sys.stderr, stderr)
            sys.stderr = stderr
        else:
            logging.getLogger().handlers.append(handler)
        logging.getLogger().addFilter(handler)
        logging.info("Continuing to log in %s", file_name)

    @staticmethod
    def duplicate_all_logging_to_mongo(addr, docid, nodeid):
        handler = MongoLogHandler(addr=addr, docid=docid, nodeid=nodeid)
        logging.info("Saving logs to Mongo on %s", addr)
        logging.getLogger().addHandler(handler)

    def change_log_message(self, msg):
        return msg

    def msg_changeable(fn):
        def msg_changeable_wrapper(self, msg, *args, **kwargs):
            msg = self.change_log_message(msg)
            return fn(self, msg, *args, **kwargs)

        msg_changeable_wrapper.__name__ = fn.__name__ + "_msg_changeable"
        return msg_changeable_wrapper

    @msg_changeable
    def log(self, level, msg, *args, **kwargs):
        self.logger.log(
            level, msg, *args, extra={"caller": self.logger.findCaller()},
            **kwargs)

    @msg_changeable
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    @msg_changeable
    def info(self, msg, *args, **kwargs):
        self.logger.info(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    @msg_changeable
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    @msg_changeable
    def error(self, msg, *args, **kwargs):
        self.logger.error(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    @msg_changeable
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    @msg_changeable
    def exception(self, msg="Exception", *args, **kwargs):
        self.logger.exception(
            msg, *args, extra={"caller": self.logger.findCaller()}, **kwargs)

    msg_changeable = staticmethod(msg_changeable)

    def event(self, name, etype, **info):
        """
        Records an event to MongoDB. Events can be later viewed in web status.
        Parameters:
            name: the name of the event, for example, "Work".
            etype: the type of the event, can be either "begin", "end" or
            "single".
            info: any extra event attributes.
        """
        if etype not in ("begin", "end", "single"):
            raise ValueError("Event type must any of the following: 'begin', "
                             "'end', 'single'")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, MongoLogHandler):
                data = {"session": handler.log_id,
                        "instance": handler.node_id,
                        "time": time.time(),
                        "domain": self.__class__.__name__,
                        "name": name,
                        "type": etype}
                dupkeys = set(data.keys()).intersection(set(info.keys()))
                if len(dupkeys) > 0:
                    raise ValueError("Event kwargs may not contain %s" %
                                     dupkeys)
                data.update(info)
                handler.events.insert(data, w=0)


class MongoLogHandler(logging.Handler):
    def __init__(self, addr, docid, nodeid, level=logging.NOTSET):
        super(MongoLogHandler, self).__init__(level)
        self._client = MongoClient("mongodb://" + addr)
        self._db = self._client.veles
        self._collection = self._db.logs
        self._events = self._db.events
        self._log_id = docid
        self._node_id = nodeid

    @property
    def log_id(self):
        return self._log_id

    @property
    def node_id(self):
        return self._node_id

    @property
    def events(self):
        return self._events

    def emit(self, record):
        data = copy(record.__dict__)
        for bs in ("levelno", "args", "msg", "module", "msecs", "processName"):
            del data[bs]
        if "caller" in data:
            data["pathname"], data["lineno"], data["funcName"], _ = \
                data["caller"]
            del data["caller"]
        data["session"] = self.log_id
        data["node"] = self.node_id
        data["pathname"] = os.path.normpath(data["pathname"])
        if os.path.isabs(data["pathname"]):
            data["pathname"] = os.path.relpath(data["pathname"], __path__)
        if data["exc_info"] is not None:
            data["exc_info"] = repr(data["exc_info"])
        try:
            self._collection.insert(data, w=0)
        except bson.errors.InvalidDocument:
            raise Bug("bson failed to encode %s" % data)
