"""
Created on Jan 21, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import print_function
import argparse
import copy
import functools
import logging
import signal
import six
from six.moves import queue
from six import add_metaclass
import sys
import threading
from traceback import print_stack
import types
from twisted.internet import reactor
from twisted.python import threadpool
import weakref

import veles.logger as logger
from veles.cmdline import CommandLineArgumentsRegistry
from veles.compat import from_none


sysexit_initial = None
exit_initial = None
quit_initial = None


class classproperty(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


def errback(failure, thread_pool=None):
    if reactor.running:
        reactor.callFromThread(failure.raiseException)
    else:
        failure.printTraceback()
        thread_pool.shutdown()


@add_metaclass(CommandLineArgumentsRegistry)
class ThreadPool(threadpool.ThreadPool, logger.Logger):
    """
    Pool of threads.
    """

    sigint_initial = None
    pools = None
    _manhole = None
    _sigint_printed = False
    _can_start = True

    def __init__(self, minthreads=2, maxthreads=1024, queue_size=2048,
                 name=None):
        """
        Initializes a new thread pool.
        """
        self.on_shutdowns = set()
        self.on_thread_enters = set()
        self.on_thread_exits = set()
        if six.PY3:
            super(ThreadPool, self).__init__(
                minthreads=minthreads, maxthreads=maxthreads, name=name)
        else:
            threadpool.ThreadPool.__init__(
                self, minthreads=minthreads, maxthreads=maxthreads, name=name)
        logger.Logger.__init__(self)
        self.q = queue.Queue(queue_size)
        self.silent = False
        self._shutting_down = False
        self._shutting_down_lock_ = threading.Lock()
        self._not_paused = threading.Event()
        self._not_paused.set()

        if ThreadPool.pools is None:
            # Initialize for the very first time
            ThreadPool.pools = []
            global sysexit_initial
            sysexit_initial = sys.exit
            sys.exit = ThreadPool.exit
            if not sys.__stdin__.closed and sys.__stdin__.isatty():
                global exit_initial
                global quit_initial
                try:
                    __IPYTHON__  # pylint: disable=E0602
                    from IPython.core.autocall import ExitAutocall
                    exit_initial = ExitAutocall.__call__
                    ExitAutocall.__call__ = ThreadPool.builtin_exit
                except NameError:
                    try:
                        import builtins
                        exit_initial = builtins.exit
                        quit_initial = builtins.quit
                        builtins.exit = ThreadPool.builtin_exit
                        builtins.quit = ThreadPool.builtin_quit
                    except:
                        pass
            ThreadPool.sigint_initial = \
                signal.signal(signal.SIGINT, ThreadPool.sigint_handler)
            assert ThreadPool.sigint_initial != ThreadPool.sigint_handler
            if not ThreadPool.manhole:
                signal.signal(signal.SIGUSR1, ThreadPool.sigusr1_handler)
            else:
                from veles.external import manhole
                manhole.install(patch_fork=False)
            signal.signal(signal.SIGUSR2, ThreadPool.sigusr2_handler)
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
            args, _ = parser.parse_known_args()
            cls._manhole = args.manhole
        return cls._manhole

    def callInThreadWithCallback(self, onResult, func, *args, **kw):
        """
        Overrides threadpool.ThreadPool.callInThreadWithCallback().
        """
        self._not_paused.wait()
        with self._shutting_down_lock_:
            if self._shutting_down or not self.started:
                return
            super(ThreadPool, self).callInThreadWithCallback(
                functools.partial(self._on_result, onResult),
                func, *args, **kw)

    def start(self):
        if not self._can_start:
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

    def _on_result(self, original, success, result):
        if original is not None:
            return original(success, result)
        if not success:
            errback(result, self)

    def _worker(self):
        """
        Overrides threadpool.ThreadPool._worker().
        """
        for on_thread_enter in self.on_thread_enters:
            on_thread_enter()
        super(ThreadPool, self)._worker()
        for on_thread_exit in self.on_thread_exits:
            on_thread_exit()

    def register_on_thread_enter(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed just after the new thread created in the thread pool.
        """
        self.on_thread_enters.add(weakref.ref(func) if weak else func)

    def register_on_thread_exit(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed just before the thread terminates.
        """
        self.on_thread_exits.add(weakref.ref(func) if weak else func)

    def register_on_shutdown(self, func, weak=True):
        """
        Adds the specified function to the list of callbacks which are
        executed before shutting down the thread pool.
        It is useful when an infinite event loop is executed in a separate
        thread and a graceful shutdown is desired. Then on_shutdown() function
        shall terminate that loop using the corresponding foreign API.
        """
        self.on_shutdowns.add(weakref.ref(func) if weak else func)

    @staticmethod
    def _put(self, item):
        """
        Private method used by shutdown() to redefine Queue's _put().
        """
        self.queue.appendleft(item)

    def shutdown(self, execute_remaining=True, force=False, timeout=0.25):
        """Safely brings thread pool down.
        """
        with self._shutting_down_lock_:
            if self not in ThreadPool.pools or self._shutting_down:
                return
            self._shutting_down = True
        self._not_paused.set()
        sdl = len(self.on_shutdowns)
        self.debug("Running %d shutdown-ers", sdl)
        skipped = 0
        for ind, on_shutdown in enumerate(self.on_shutdowns):
            if isinstance(on_shutdown, weakref.ReferenceType):
                on_shutdown = on_shutdown()
                if on_shutdown is None:
                    # The weakly referenced object no longer exists
                    skipped += 1
                    continue
            self.debug("%d/%d - %s", ind, sdl, str(on_shutdown))
            try:
                on_shutdown()
            except:
                self.exception("Ignored the following exception in shutdowner "
                               "%s:", on_shutdown)
        self.debug("Skipped %d dead refs. Requesting threads to quit", skipped)
        self.on_shutdowns.clear()
        self.joined = True
        self.started = False
        threads = copy.copy(self.threads)
        if not execute_remaining:
            self.q._put = types.MethodType(ThreadPool._put, self.q)
        while self.workers:
            self.q.put(threadpool.WorkerStop)
            self.workers -= 1
        self.debug("Joining threads")
        quant = timeout / 10
        for thread in threads:
            if threading.current_thread() == thread:
                continue
            attempts = 1
            thread.join(quant)
            while thread.is_alive() and attempts < 10:
                if self.q.empty():
                    self.q.put_nowait(threadpool.WorkerStop)
                thread.join(quant)
                attempts += 1
            if force and thread.is_alive():
                if not self.silent:
                    self.warning("Stack trace of probably deadlocked #%d:",
                                 thread.ident)
                    print_stack(sys._current_frames()[thread.ident],
                                file=sys.stdout)
                self.force_thread_to_stop(thread)
                if not self.silent:
                    self.warning(
                        "Failed to join with thread #%d since the  timeout"
                        " (%.2f sec) was exceeded.%s",
                        thread.ident, timeout, " It was killed."
                        if ThreadPool.thread_can_be_forced_to_stop(thread)
                        else " It was not killed due to the lack of _stop "
                        "in Thread class from Python's stdlib.")
        ThreadPool.pools.remove(self)
        if not len(ThreadPool.pools):
            global sysexit_initial
            sys.exit = sysexit_initial

        if not sys.__stdin__.closed and sys.__stdin__.isatty():
            global exit_initial
            global quit_initial
            try:
                __IPYTHON__  # pylint: disable=E0602
                from IPython.core.autocall import ExitAutocall
                ExitAutocall.__call__ = exit_initial
            except NameError:
                try:
                    import builtins
                    builtins.exit = exit_initial
                    builtins.quit = quit_initial
                except:
                    pass
        self.debug("%s was shutted down", repr(self))
        self._shutting_down = False

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
        ThreadPool._can_start = False
        pools = copy.copy(ThreadPool.pools)
        logging.getLogger("ThreadPool").debug(
            "Shutting down %d pools...", len(pools))
        for pool in pools:
            pool.shutdown(execute_remaining, force, timeout)

    @staticmethod
    def _exit():
        ThreadPool.shutdown_pools()
        ThreadPool.debug_deadlocks()

    @staticmethod
    def exit(retcode=0):
        """
        Terminates the running program safely.
        """
        ThreadPool._exit()
        if sys.exit == ThreadPool.exit:
            print("Detected an infinite recursion in sys.exit(), "
                  "restoring %s" % sysexit_initial,
                  file=sys.stderr)
            assert sysexit_initial != ThreadPool.exit
            sys.exit = sysexit_initial
        sys.exit(retcode)

    @staticmethod
    def builtin_exit(*args, **kwargs):
        """
        Terminates the interactive shell safely.
        """
        ThreadPool._exit()
        exit_initial(*args, **kwargs)

    @staticmethod
    def builtin_quit(*args, **kwargs):
        """
        Terminates the interactive shell safely.
        """
        ThreadPool._exit()
        quit_initial(*args, **kwargs)

    @staticmethod
    def _warn_about_sigint_hysteria(log):
        log.warning(
            "Please, stop hitting Ctrl-C hysterically and let me "
            "die peacefully.\nThis will not anticipate the "
            "program's exit because currently Python is trying to "
            "join all the threads\nand some of them can be very "
            "busy on the native side, e.g. running some sophisticated "
            "OpenCL code.")

    @staticmethod
    def sigint_handler(sign, frame):
        """
        Private method - handler for SIGINT.
        """
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
                if not ThreadPool._sigint_printed:
                    log.warning("Raising KeyboardInterrupt since "
                                "Twisted reactor is not running")
                    ThreadPool._sigint_printed = True
                    raise from_none(KeyboardInterrupt())
                ThreadPool._warn_about_sigint_hysteria(log)
            else:
                if not ThreadPool._sigint_printed:
                    log.critical("KeyboardInterrupt")
                    ThreadPool.debug_deadlocks()
                    ThreadPool._sigint_printed = True
                else:
                    ThreadPool._warn_about_sigint_hysteria(log)

    @staticmethod
    def sigusr1_handler(sign, frame):
        """
        Private method - handler for SIGUSR1.
        """
        print("SIGUSR1 was received, dumping the current frames...")
        ThreadPool.print_thread_stacks()

    @staticmethod
    def sigusr2_handler(sign, frame):
        print("SIGUSR2 was received")

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

    @staticmethod
    def debug_deadlocks():
        if threading.active_count() > 1:
            if threading.active_count() == 2:
                for thread in threading.enumerate():
                    if (thread.name.startswith('timeout') or
                            thread.name == "IPythonHistorySavingThread"):
                        # veles.tests.timeout registers atexit
                        return
            logging.warning("There are currently more than 1 threads still "
                            "running. A deadlock is likely to happen.\n%s",
                            str(threading.enumerate())
                            if hasattr(threading, "_active")
                            else "<unable to list active threads>")
            ThreadPool.print_thread_stacks()
