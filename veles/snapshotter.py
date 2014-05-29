"""
Created on May 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import bz2
import gzip
import lzma
import os
import six
import sys
import time
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
from veles.pickle2 import pickle
from veles.units import Unit, IUnit


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622


@implementer(IUnit)
class SnapshotterBase(Unit, TriviallyDistributable):
    """Base class for various data exporting units.

    Defines:
        file_name - the file name of the last snapshot
        time - the time of the last snapshot

    Must be defined before initialize():
        suffix - the file name suffix where to take snapshots

    Attributes:
        compress - the compression applied to pickles: None or '', gz, bz2, xz
        compress_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocation number
        time_interval - take no more than one snapshot within this time window
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(SnapshotterBase, self).__init__(workflow, **kwargs)
        self.directory = kwargs.get("directory", "/tmp")
        self.prefix = kwargs.get("prefix", "")
        self.compress = kwargs.get("compress", "gz")
        self.compress_level = kwargs.get("compress_level", 9)
        self.interval = kwargs.get("interval", 1)
        self.time_interval = kwargs.get("time_interval", 1)
        self.time = 0
        self._skipped_counter = 0
        self.file_name = ""
        self.suffix = None

    def initialize(self, **kwargs):
        self.time = time.time()

    def run(self):
        self._skipped_counter += 1
        if self._skipped_counter < self.interval:
            return
        if time.time() - self.time < self.time_interval:
            return
        self.export()
        self.time = time.time()

    def export(self):
        """This method should be overridden in inherited classes.
        """
        pass


class Snapshotter(SnapshotterBase):
    """Takes workflow snapshots.

    Defines:
        file_name - the file name of the last snapshot
        time - the time of the last snapshot

    Must be defined before initialize():
        suffix - the file name suffix where to take snapshots

    Attributes:
        compress - the compression applied to pickles: None or '', gz, bz2, xz
        compress_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocation number
        time_interval - take no more than one snapshot within this time window
    """

    CODECS = {
        None: lambda n, l: open(n, "wb"),
        "": lambda n, l: open(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(n, "wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    def export(self):
        ext = ("." + self.compress) if self.compress else ""
        rel_file_name = "%s_%s.%d.pickle%s" % (
            self.prefix, self.suffix, 3 if six.PY3 else 2, ext)
        self.file_name = os.path.join(self.directory, rel_file_name)
        with self._open_file() as fout:
            pickle.dump(self.workflow, fout)
        logged = set()
        for u in self.workflow._units:
            if (hasattr(u, "weights") and hasattr(u.weights, "mem")
                and u.weights.mem is not None
                and id(u.weights.mem) not in logged):
                self.info("%s: Weights range: [%.6f, %.6f]",
                          u.__class__.__name__,
                          u.weights.mem.min(), u.weights.mem.max())
                logged.add(id(u.weights.mem))
            if (hasattr(u, "bias") and hasattr(u.bias, "mem")
                and u.bias.mem is not None
                and id(u.bias.mem) not in logged):
                self.info("%s: Bias range: [%.6f, %.6f]",
                          u.__class__.__name__,
                          u.bias.mem.min(), u.bias.mem.max())
                logged.add(id(u.bias.mem))
            if (hasattr(u, "output") and hasattr(u.output, "mem")
                and u.output.mem is not None
                and id(u.output.mem) not in logged):
                self.info("%s: Output range: [%.6f, %.6f]",
                          u.__class__.__name__,
                          u.output.mem.min(), u.output.mem.max())
                logged.add(id(u.output.mem))
        self.info("Wrote %s" % self.file_name)
        file_name_link = os.path.join(
            self.directory, "%s_current.%d.pickle%s" % (
                self.prefix, 3 if six.PY3 else 2, ext))
        if os.path.exists(file_name_link):
            os.remove(file_name_link)
        os.symlink(rel_file_name, file_name_link)

    def _open_file(self):
        return Snapshotter.CODECS[self.compress](self.file_name,
                                                 self.compress_level)
