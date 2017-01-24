# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jan 21, 2014

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


from __future__ import print_function
import argparse
from copy import copy
import functools
import logging
import signal
import six
from six import add_metaclass
import sys
import threading
from traceback import print_stack
import types
from twisted.internet import reactor
from twisted.python import threadpool
import weakref

from veles.external import manhole
import veles.logger as logger
from veles.cmdline import CommandLineArgumentsRegistry, classproperty
from veles.compat import from_none, is_interactive


def errback(failure, thread_pool=None):
    if reactor.running:
        reactor.callFromThread(failure.raiseException)
    else:
        tmap = {thr.ident: thr.name for thr in threading.enumerate()}
        thread_pool.error("Unhandled error inside %s:\n%s",
                          tmap[threading.current_thread().ident],
                          failure.getTraceback())
        thread_pool.failure = failure
        thread_pool.shutdown(execute_remaining=False, force=True)


@add_metaclass(CommandLineArgumentsRegistry)
class ThreadPool(threadpool.ThreadPool, logger.Logger):
    """
    Pool of threads.
    """

    sigint_initial = None
    sysexit_initial = None
    exit_initial = None
    quit_initial = None
    pools = None
    atexits = set()
    interrupted = False
    _manhole = None
    sigint_printed = False

    def __init__(self, minthreads=2, maxthreads=1024,
                 name=None, workflow=None):
        """
        Initializes a new thread pool.
        """
        self.on_shutdowns = set()
        self.on_thread_enters = set()
        self.on_thread_exits = set()
        if name is None:
            name = str(id(self))
        name += str(len(ThreadPool.pools) + 1) \
            if ThreadPool.pools is not None else "1"
        if six.PY3:
            super(ThreadPool, self).__init__(
                minthreads=minthreads, maxthreads=maxthreads, name=name)
        else:
            threadpool.ThreadPool.__init__(
                self, minthreads=minthreads, maxthreads=maxthreads, name=name)
        logger.Logger.__init__(self)
        # remove queue_size, because it is not allowed to set q attribute
        # in new version of threadpool.ThreadPool
        # self.q = queue.Queue(queue_size)
        # TODO(lyubov.p) get better fix
        self.silent = False
        self._dead = False
        self._stopping = False
        self._lock = threading.Lock()
        self._not_paused = threading.Event()
        self._not_paused.set()
        self.failure = None
        self.workflow = workflow

        if ThreadPool.pools is None:
            # Initialize for the very first time
            ThreadPool.pools = []
            ThreadPool.sysexit_initial = sys.exit
            sys.exit = ThreadPool.exit
            ThreadPool.setup_interactive_exit()
            ThreadPool.sigint_initial = \
                signal.signal(signal.SIGINT, ThreadPool.sigint_handler)
            assert ThreadPool.sigint_initial != ThreadPool.sigint_handler
            if not ThreadPool.manhole:
                signal.signal(signal.SIGUSR1, self.sigusr1_handler)
                signal.signal(signal.SIGUSR2, self.sigusr2_handler)
        ThreadPool.pools.append(self)

    def __del__(self):
        if not self.joined:
            self.shutdown(False, True)

    @staticmethod
    def init_parser(**kwargs):
        parser = kwargs.get("parser", argparse.ArgumentParser())
        parser.add_argument("--manhole", default=False,
                            help="run the embedded interactive shell "
                            "accessible through a UNIX socket",
                            action='store_true')
        return parser

    @classproperty
    def manhole(cls):
        if cls._manhole is None:
            parser = cls.init_parser()
            args, _ = parser.parse_known_args(cls.class_argv)
            cls._manhole = args.manhole
        return cls._manhole

    @property
    def workflow(self):
        return self._workflow()

    @workflow.setter
    def workflow(self, value):
        self._workflow = weakref.ref(value) if value is not None else None
        if value is not None and self.manhole:
            self._install_manhole()

    def _install_manhole(self):
        manhole.install(patch_fork=False, workflow=self.workflow)

    def callInThreadWithCallback(self, onResult, func, *args, **kw):
        """
        Overrides threadpool.ThreadPool.callInThreadWithCallback().
        """
        self._not_paused.wait()
        if self._stopping:
            return
        with self._lock:
            if self._dead:
                return
            super(ThreadPool, self).callInThreadWithCallback(
                functools.partial(self._on_result, onResult),
                func, *args, **kw)

    def start(self):
        if self._stopping:
            return
        with self._lock:
            if self._dead:
                return
            super(ThreadPool, self).start()
        self.debug("ThreadPool with %d threads has been started",
                   len(self.threads))

    def pause(self):
        self._not_paused.clear()
        self.info("ThreadPool with %d threads has been suspended",
                  len(self.threads))

    def resume(self):
        self._not_paused.set()
        self.info("ThreadPool with %d threads has been resumed",
                  len(self.threads))

    @property
    def paused(self):
        return not self._not_paused.is_set()

    def register_on_thread_enter(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed just after the new thread created in the thread pool.
        """
        self._add_callback(self.on_thread_enters, func, weak)

    def unregister_on_thread_enter(self, func, warn=True):
        """
        Removes the specified function from the list of callbacks which are
        executed just after the new thread created in the thread pool.
        """
        self._remove_callback(self.on_thread_enters, func, warn)

    def register_on_thread_exit(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed just before the thread terminates.
        """
        self._add_callback(self.on_thread_exits, func, weak)

    def unregister_on_thread_exit(self, func, warn=True):
        """
        Removes the specified function from the list of callbacks which are
        executed just before the thread terminates.
        """
        self._remove_callback(self.on_thread_exits, func, warn)

    def register_on_shutdown(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed before shutting down the thread pool.
        It is useful when an infinite event loop is executed in a separate
        thread and a graceful shutdown is desired. Then on_shutdown() function
        shall terminate that loop using the corresponding foreign API.
        """
        self._add_callback(self.on_shutdowns, func, weak)

    def unregister_on_shutdown(self, func, warn=True):
        self._remove_callback(self.on_shutdowns, func, warn)

    @staticmethod
    def register_atexit(func, weak=True):
        ThreadPool._add_callback(ThreadPool.atexits, func, weak)

    def stop(self, execute_remaining=True, force=False, timeout=1.0):
        self._stopping_call(self._stop, execute_remaining, force, timeout)

    def shutdown(self, execute_remaining=True, force=False, timeout=1.0):
        self._stopping_call(self._shutdown, execute_remaining, force, timeout)

    def _on_result(self, original, success, result):
        if original is not None:
            return original(success, result)
        if not success:
            errback(result, self)

    def _worker(self):
        """
        Overrides threadpool.ThreadPool._worker().
        """
        for on_thread_enter in self._iter_weak(self.on_thread_enters):
            on_thread_enter()
        try:
            super(ThreadPool, self)._worker()
        finally:
            for on_thread_exit in self._iter_weak(self.on_thread_exits):
                on_thread_exit()

    @staticmethod
    def _iter_weak(iterable):
        for obj in copy(iterable):
            if isinstance(obj, weakref.ReferenceType):
                obj = obj()
                if obj is not None:
                    yield obj
            else:
                yield obj

    @staticmethod
    def _add_callback(cont, func, weak):
        cont.add(weakref.ref(func) if weak else func)

    def _remove_callback(self, cont, func, warn):
        if func in cont:
            cont.remove(func)
            return
        ref = weakref.ref(func)
        if ref in cont:
            cont.remove(ref)
            return
        refhash = hash(ref)
        for key in set(cont):
            if hash(key) == refhash and \
                    isinstance(key, weakref.ReferenceType) and key() is None:
                cont.remove(key)
                break
        else:
            if not warn:
                return
            self.warning("remove_callback: %s was not found in %s", func, cont)

    @staticmethod
    def _put(self, item):
        """
        Private method used by shutdown() to redefine Queue's _put().
        """
        self.queue.appendleft(item)

    def _stopping_call(self, method, *args, **kwargs):
        if self._stopping:
            return
        with self._lock:
            self._stopping = True
        with self._lock:
            method(*args, **kwargs)
            self._stopping = False

    def _stop(self, execute_remaining, force, timeout):
        self._not_paused.set()
        self.started = False
        self._stopping = True
        threads = copy(self.threads)
        if not execute_remaining:
            self.q._put = types.MethodType(ThreadPool._put, self.q)
        for _ in range(self.workers):
            self.q.put(threadpool.WorkerStop)
        tmap = {thr.ident: thr.name for thr in threading.enumerate()}
        self.debug("Joining %d threads", self.workers)
        for thread in threads:
            if threading.current_thread() == thread:
                continue
            thread.join(timeout)
            if thread.is_alive():
                if not self.silent:
                    self.warning("Stack trace of probably deadlocked #%d:",
                                 thread.ident)
                    print_stack(sys._current_frames()[thread.ident],
                                file=sys.stdout)
                if force:
                    self.force_thread_to_stop(thread)
                    if not self.silent:
                        self.warning(
                            "Failed to join with thread #%d since the  timeout"
                            " (%.2f sec) was exceeded.%s",
                            thread.ident, timeout, " It was killed."
                            if ThreadPool.thread_can_be_forced_to_stop(thread)
                            else " It was not killed due to the lack of _stop "
                                 "in Thread class from Python's stdlib.")
                else:
                    self.warning("Failed to join %s", tmap[thread.ident])
            else:
                self.workers -= 1
        self.joined = True

    def _shutdown(self, execute_remaining, force, timeout):
        """Safely brings thread pool down.
        """
        if self not in ThreadPool.pools or self._dead:
            return
        self._dead = True
        sdl = len(self.on_shutdowns)
        self.debug("Running %d shutdown-ers", sdl)
        skipped = 0
        for ind, on_shutdown in enumerate(self._iter_weak(self.on_shutdowns)):
            if on_shutdown is None:
                # The weakly referenced object no longer exists
                skipped += 1
                continue
            self.debug("%d/%d - %s", ind + 1, sdl, str(on_shutdown))
            try:
                on_shutdown()
            except:
                self.exception("Ignored the following exception in shutdowner "
                               "%s:", on_shutdown)
        self.debug("Skipped %d dead refs", skipped)
        self.on_shutdowns.clear()
        self._stop(execute_remaining, force, timeout)
        ThreadPool.pools.remove(self)
        if not len(ThreadPool.pools):
            sys.exit = ThreadPool.__dict__["sysexit_initial"]
        self.debug("%s was shutted down", repr(self))

    @staticmethod
    def thread_can_be_forced_to_stop(thread):
        return hasattr(thread, "_stop") and callable(thread._stop) and \
            thread != threading.main_thread() and thread.is_alive()

    def force_thread_to_stop(self, thread):
        if ThreadPool.thread_can_be_forced_to_stop(thread):
            try:
                thread._stop()
            except:
                if not self.silent:
                    self.warning("Failed to kill %s", str(thread))

    @staticmethod
    def shutdown_pools(execute_remaining=True, force=False, timeout=0.25):
        """
        Private method to shut down all the pools.
        """
        if ThreadPool.pools is None:
            return
        pools = copy(ThreadPool.pools)
        logging.getLogger("ThreadPool").debug(
            "Shutting down %d pools...", len(pools))
        for pool in pools:
            pool.shutdown(execute_remaining, force, timeout)

    @staticmethod
    def run_atexits():
        for atexit in ThreadPool._iter_weak(ThreadPool.atexits):
            atexit()

    @staticmethod
    def _exit():
        ThreadPool.shutdown_pools()
        ThreadPool.debug_deadlocks()
        ThreadPool.run_atexits()

    @staticmethod
    def exit(retcode=0):
        """
        Terminates the running program safely.
        """
        ThreadPool._exit()
        sysexit_initial = ThreadPool.__dict__["sysexit_initial"]
        if sys.exit == ThreadPool.exit:
            assert sysexit_initial != ThreadPool.exit
            sys.exit = sysexit_initial
            logging.getLogger("ThreadPool").warning(
                "Detected an infinite recursion in sys.exit(), restored %s",
                sysexit_initial)
        sys.exit(retcode)

    @staticmethod
    def reset():
        ThreadPool.interrupted = False
        ThreadPool.sigint_printed = False

    @staticmethod
    def builtin_exit(*args, **kwargs):
        """
        Terminates the interactive shell safely.
        """
        ThreadPool._exit()
        ThreadPool.exit_initial(*args, **kwargs)

    @staticmethod
    def builtin_quit(*args, **kwargs):
        """
        Terminates the interactive shell safely.
        """
        ThreadPool._exit()
        ThreadPool.quit_initial(*args, **kwargs)

    @staticmethod
    def _ipython_ask_exit(*args, **kwargs):
        ThreadPool._exit()
        ThreadPool._ipython_ask_exit_initial(*args, **kwargs)

    @staticmethod
    def _warn_about_sigint_hysteria(log):
        import os
        log.warning(
            "Please, stop hitting Ctrl-C hysterically and let me "
            "die peacefully.\nThis will not anticipate the "
            "program's exit because currently Python is trying to "
            "join all the threads\nand some of them can be very "
            "busy on the native side, e.g. running some sophisticated "
            "OpenCL code. Execute kill -SIGUSR1 %d to know which.",
            os.getpid())

    @staticmethod
    def _warn_about_sigint_interactive_reactor(log):
        log.warning(
            "We are running in interactive mode, so Twisted reactor runs in a "
            "separate thread. KeyboardInterrupt are not delivered to it to "
            "enable further workflow runs (Twisted reactor is not restartable)"
            ", so if you wish to nuke it, explicitly call reactor.stop() or "
            "launcher.stop_reactor().")

    @staticmethod
    def sigint_handler(sign, frame):
        """
        Private method - handler for SIGINT.
        """
        ThreadPool.interrupted = True
        ThreadPool.shutdown_pools(execute_remaining=False, force=True)
        log = logging.getLogger("ThreadPool")
        try:
            # ThreadPool.sigint_initial(sign, frame) does not work on Python 2
            sigint_initial = ThreadPool.__dict__['sigint_initial']
            if sigint_initial == ThreadPool.sigint_handler:
                log.warning("Prevented an infinite recursion: sigint_initial")
            else:
                sigint_initial(sign, frame)
        except KeyboardInterrupt:
            if not reactor.running:
                if not ThreadPool.sigint_printed:
                    log.warning("Raising KeyboardInterrupt since "
                                "Twisted reactor is not running")
                    ThreadPool.sigint_printed = True
                    raise from_none(KeyboardInterrupt())
                ThreadPool._warn_about_sigint_hysteria(log)
            else:
                if not ThreadPool.sigint_printed:
                    log.critical("KeyboardInterrupt")
                    ThreadPool.debug_deadlocks()
                    ThreadPool.sigint_printed = True
                else:
                    if not is_interactive():
                        ThreadPool._warn_about_sigint_hysteria(log)
                    else:
                        ThreadPool._warn_about_sigint_interactive_reactor(log)

    def sigusr1_handler(self, sign, frame):
        """
        Private method - handler for SIGUSR1.
        """
        self.warning("SIGUSR1 was received, dumping the current frames...")
        ThreadPool.print_thread_stacks()

    def sigusr2_handler(self, sign, frame):
        if self.workflow is not None:
            self.warning("SIGUSR2 was received, installing the manhole...")
            self._install_manhole()
        else:
            self.warning("SIGUSR2 was received, unable to install the manhole "
                         "because no workflow's been registered")

    @staticmethod
    def print_thread_stacks():
        if not hasattr(sys, "_current_frames"):
            print("Threads' stacks printing is not implemented for this "
                  "Python interpreter", file=sys.stderr)
            return
        tmap = {thr.ident: thr.name for thr in threading.enumerate()}
        for tid, stack in sys._current_frames().items():
            print("-" * 80)
            print("Thread #%d (%s):" % (tid, tmap.get(tid, "<unknown>")))
            print_stack(stack, file=sys.stdout)
        sys.stdout.flush()

    KNOWN_RUNNING_THREADS = {"IPythonHistorySavingThread", "TwistedReactor",
                             "MainThread", "twisted.internet.reactor",
                             "test_timeout", "Manhole"}

    @staticmethod
    def debug_deadlocks():
        if threading.active_count() > 1:
            for thread in threading.enumerate():
                if any(name in thread.name
                       for name in ThreadPool.KNOWN_RUNNING_THREADS):
                    # veles.tests.timeout registers atexit
                    continue
                break
            else:
                return
            logging.warning("There are currently more than 1 threads still "
                            "running. A deadlock is likely to happen.\n%s",
                            str(threading.enumerate())
                            if hasattr(threading, "_active")
                            else "<unable to list active threads>")
            ThreadPool.print_thread_stacks()

    @staticmethod
    def setup_interactive_exit():
        if not is_interactive():
            return
        try:
            __IPYTHON__  # pylint: disable=E0602
            from IPython.terminal.interactiveshell import InteractiveShell
            shell = InteractiveShell.instance()
            ThreadPool.exit_initial = shell.exiter.__call__
            shell.exiter.__call__ = ThreadPool.builtin_exit
            ThreadPool._ipython_ask_exit_initial = shell.ask_exit
            shell.ask_exit = ThreadPool._ipython_ask_exit
        except NameError:
            try:
                import builtins
                ThreadPool.exit_initial = builtins.exit
                ThreadPool.quit_initial = builtins.quit
                builtins.exit = ThreadPool.builtin_exit
                builtins.quit = ThreadPool.builtin_quit
            except:
                pass

    @staticmethod
    def restore_initial_interactive_exit():
        if not is_interactive():
            return
        try:
            __IPYTHON__  # pylint: disable=E0602
            from IPython.terminal.interactiveshell import InteractiveShell
            shell = InteractiveShell.instance()
            shell.exiter.__call__ = ThreadPool.exit_initial
            shell.ask_exit = ThreadPool._ipython_ask_exit_initial
        except NameError:
            try:
                import builtins
                builtins.exit = ThreadPool.exit_initial
                builtins.quit = ThreadPool.quit_initial
            except:
                pass
