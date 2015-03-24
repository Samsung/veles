# encoding: utf-8
# progressbar  - Text progress bar library for Python.
# Copyright (c) 2005 Nilton Volpato
#           (c) 2015 Samsung Electronics Co., Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Prints a spinning dash every once and a while.
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
