"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 21, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import os
import signal
from six import print_
import sys
import threading
import unittest
from veles.compat import from_none

from veles.thread_pool import ThreadPool


new_event = threading.Event()
thread_args = (None, None)


def wait():
    while new_event.wait():
        new_event.clear()
        event, seconds = thread_args
        if event is None:
            return
        if not event.wait(seconds):
            print_("Timeout %.1f sec - sending SIGINT and praying for all "
                   "threads to join." % seconds, file=sys.stderr)
            ThreadPool.interrupted = True
            os.kill(os.getpid(), signal.SIGINT)
            # SIGTERM is useless in most real world scenarious with nosetests

thread = threading.Thread(target=wait, name='test_timeout')
sysexit = sys.exit


def interrupt_waiter():
    global thread_args
    event, _ = thread_args
    thread_args = (None, None)
    if event is not None:
        event.set()
    new_event.set()
    if thread.is_alive():
        thread.join()


def sigint_handler(sign, frame):
    interrupt_waiter()
    sigint_initial(sign, frame)


sigint_initial = signal.signal(signal.SIGINT, sigint_handler)


def shutdown(errcode=0):
    interrupt_waiter()
    if sysexit == ThreadPool.exit and sys.exit == shutdown:
        print_("timeout sysexit <-> ThreadPool.exit recursion",
               file=sys.stderr)
        os._exit(errcode)
    sysexit(errcode)


sys.exit = shutdown


def timeout(value=60):
    if value > 1800:
        raise ValueError("Timeouts bigger than half an hour are useless")

    if not thread.is_alive():
        thread.start()

    def timeout_impl(fn):
        def unknown():
            pass

        name = getattr(fn, '__name__', getattr(fn, 'func', unknown).__name__)

        def wrapped(self, *args, **kwargs):
            assert isinstance(self, unittest.TestCase)
            event = threading.Event()
            global thread_args
            thread_args = (event, value)
            thread.name = 'test_timeout@%s' % name
            new_event.set()
            try:
                res = fn(self, *args, **kwargs)
            except KeyboardInterrupt:
                raise from_none(self.failureException("KeyboardInterrupt"))
            finally:
                event.set()
            return res

        wrapped.__name__ = name
        return wrapped
    return timeout_impl
