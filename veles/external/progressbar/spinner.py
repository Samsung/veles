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


__symbols = cycle(('|\r', '/\r', '—\r', '\\\r'))
__lock = Lock()
__last_flush_time = clock()


def spin(interval=0.2):
    with __lock:
        time = clock()
        global __last_flush_time
        if time - __last_flush_time < interval:
            return
        __last_flush_time = time
        stdout.write(next(__symbols))
        stdout.flush()
