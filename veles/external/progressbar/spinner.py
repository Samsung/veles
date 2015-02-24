# encoding: utf-8
"""
Created on Feb 25, 2015

Prints a spinning dash every once and a while.

Copyright (c) 2015 Samsung Electronics Co., Ltd.
"""

from itertools import cycle
from sys import stdout
from time import clock
from threading import Lock
try:
    from twisted.internet import reactor
except ImportError:
    reactor = None


__symbols = cycle(('|\r', '/\r', '—\r', '\\\r'))
__lock = Lock()
__last_flush_time = clock()


def spin(interval=0.2):
    with __lock:
        stdout.write(next(__symbols))
        if reactor is None or not reactor.running:
            time = clock()
            global __last_flush_time
            if time - __last_flush_time < interval:
                return
            stdout.flush()
            __last_flush_time = time
        else:
            reactor.callFromThread(stdout.flush)
