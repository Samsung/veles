"""
Created on May 21, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import os
import signal
from six import print_
import sys
import threading


new_event = threading.Event()
thread_args = (None, None)


def wait():
    while new_event.wait():
        new_event.clear()
        event, seconds = thread_args
        if event is None:
            return
        if not event.wait(seconds):
            print_("Timeout %.1f sec - sending SIGTERM" % seconds,
                   file=sys.stderr)
            os.kill(os.getpid(), signal.SIGTERM)


thread = threading.Thread(target=wait, name='timeout')
thread.start()
sysexit = sys.exit


def shutdown(errcode=0):
    global thread_args
    thread_args = (None, None)
    new_event.set()
    if thread.is_alive():
        thread.join()
    sysexit(errcode)


sys.exit = shutdown


def timeout(value=60):
    def timeout_impl(fn):
        def unknown():
            pass

        name = getattr(fn, '__name__', getattr(fn, 'func', unknown).__name__)

        def wrapped(*args, **kwargs):
            event = threading.Event()
            global thread_args
            thread_args = (event, value)
            thread.name = 'timeout@%s' % name
            new_event.set()
            try:
                res = fn(*args, **kwargs)
            finally:
                event.set()
            return res

        wrapped.__name__ = name
        return wrapped
    return timeout_impl
