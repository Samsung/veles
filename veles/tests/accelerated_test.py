"""
Created on March 19, 2015

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import gc
import inspect
import logging
from six import PY3
import sys
from types import FrameType
import unittest

from veles.backends import Device, BackendRegistry
from veles.config import root
from veles.dummy import DummyWorkflow
from veles.logger import Logger
if PY3:
    from veles.memory import Vector
from veles.opencl_types import dtypes


def multi_device(numpy=False):
    def real_multi_device(fn):
        def test_wrapped(self):
            backends = list(self.backends)
            if numpy:
                backends.append(None)
            for cls in backends:
                self.device = cls() if cls is not None else None
                self.info("Selected %s",
                          cls.__name__ if cls is not None else "numpy")
                self.seed()
                fn(self)
                self.parent.stopped = False

        test_wrapped.__name__ = fn.__name__
        return test_wrapped
    return real_multi_device


def assign_backend(backend):
    def wrapped(cls):
        cls.DEVICE = BackendRegistry.backends[backend]
        cls.ABSTRACT = False
        return cls

    return wrapped


if "nose" in sys.modules:
    import nose.proxy
    nose.proxy.ResultProxy.addExpectedFailure = \
        lambda this, other, *_: this.addSuccess(other)


class AcceleratedTest(unittest.TestCase, Logger):
    backends = [v for k, v in sorted(BackendRegistry.backends.items())
                if k not in ("auto", "numpy")]
    DEVICE = Device
    ABSTRACT = False

    def __init__(self, *args, **kwargs):
        Logger.__init__(self)
        unittest.TestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        self.device = self.DEVICE()
        self.parent = self.getParent()
        self._dtype = dtypes[root.common.precision_type]

    def getParent(self):
        return DummyWorkflow()

    @property
    def dtype(self):
        return self._dtype

    def debug(self, msg, *args, **kwargs):
        Logger.debug(self, msg, *args, **kwargs)

    def seed(self):
        pass

    def tearDown(self):
        if PY3:
            Vector.reset_all()
        if sys.exc_info() == (None,) * 3:
            del self.parent
        del self.device
        gc.collect()
        if PY3:
            assert len(gc.garbage) == 0, str(gc.garbage)

    def run(self, result=None):
        failure = getattr(type(self), "__unittest_expecting_failure__", False)
        for m in inspect.getmembers(type(self), predicate=inspect.isfunction):
            m[1].__unittest_expecting_failure__ = failure
        if not type(self).ABSTRACT:
            return unittest.TestCase.run(self, result)
        else:
            return None

    @staticmethod
    def main():
        Logger.setup_logging(logging.DEBUG)
        unittest.main()


def print_cycles(objects, out=sys.stdout, show_progress=False):
    """
    objects:       A list of objects to find cycles in.  It is often useful
                   to pass in gc.garbage to find the cycles that are
                   preventing some objects from being garbage collected.
    out:     The stream for output.
    show_progress: If True, print the number of objects reached as they are
                   found.
    """

    def print_path(path):
        for i, step in enumerate(path):
            # next "wraps around"
            nextobj = path[(i + 1) % len(path)]

            out.write("   %s -- " % type(step))
            if isinstance(step, dict):
                for key, val in step.items():
                    if val is nextobj:
                        out.write("[%s]" % key)
                        break
                    if key is nextobj:
                        out.write("[key] = %s" % val)
                        break
            elif isinstance(step, (list, tuple)):
                if nextobj not in step:
                    out.write(str(nextobj))
                else:
                    out.write("[%d] %s" % (step.index(nextobj), nextobj))
            else:
                out.write(repr(step))
            out.write(" ->\n")
        out.write("\n")

    def recurse(head, start, visited, current_path):
        if show_progress:
            out.write("%d\r" % len(visited))

        visited.add(id(head))

        referents = gc.get_referents(head)
        for referent in referents:
            # If we've found our way back to the start, this is
            # a cycle, so print it out
            if referent is start:
                print_path(current_path)

            # Don't go back through the original list of objects, or
            # through temporary references to the object, since those
            # are just an artifact of the cycle detector itself.
            elif referent is objects or isinstance(referent, FrameType):
                continue

            # We haven't seen this object before, so recurse
            elif id(referent) not in visited:
                recurse(referent, start, visited, current_path + [head])

    for obj in objects:
        out.write("Examining: %r\n" % obj)
        recurse(obj, obj, set(), [])
