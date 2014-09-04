"""
Created on May 16, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import bz2
import gzip
import lzma
import os
import sys
import time
from zope.interface import implementer

from veles.distributable import IDistributable
from veles.pickle2 import pickle, best_protocol
from veles.units import Unit, IUnit


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622


@implementer(IUnit, IDistributable)
class SnapshotterBase(Unit):
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

    def init_unpickled(self):
        super(SnapshotterBase, self).init_unpickled()
        self.slaves = {}

    def initialize(self, **kwargs):
        self.time = time.time()

    def run(self):
        if self.is_slave:
            return
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

    def generate_data_for_slave(self, slave):
        self.slaves[slave.id] = 1

    def generate_data_for_master(self):
        return True

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        self._slave_ended(slave)

    def _slave_ended(self, slave):
        if slave is None:
            return
        if slave.id not in self.slaves:
            return
        del self.slaves[slave.id]
        if not (len(self.slaves) or self.gate_skip or self.gate_block):
            self.run()

    def drop_slave(self, slave):
        if slave.id in self.slaves:
            self._slave_ended(slave)


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

    WRITE_CODECS = {
        None: lambda n, l: open(n, "wb"),
        "": lambda n, l: open(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(n, "wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    READ_CODECS = {
        ".pickle": lambda name: open(name, "rb"),
        ".gz": lambda name: gzip.GzipFile(name, "rb"),
        ".bz2": lambda name: bz2.BZ2File(name, "rb"),
        ".xz": lambda name: lzma.LZMAFile(name, "rb")
    }

    def export(self):
        ext = ("." + self.compress) if self.compress else ""
        rel_file_name = "%s_%s.%d.pickle%s" % (
            self.prefix, self.suffix, best_protocol, ext)
        self.file_name = os.path.join(self.directory, rel_file_name)
        self.debug("Snapshotting...")
        with self._open_file() as fout:
            pickle.dump(self.workflow, fout, protocol=best_protocol)
        self.info("Snapshotted to %s" % self.file_name)
        file_name_link = os.path.join(
            self.directory, "%s_current.%d.pickle%s" % (
                self.prefix, best_protocol, ext))
        if os.path.exists(file_name_link):
            os.remove(file_name_link)
        os.symlink(rel_file_name, file_name_link)

    def _open_file(self):
        return Snapshotter.WRITE_CODECS[self.compress](self.file_name,
                                                       self.compress_level)

    @staticmethod
    def import_(file_name):
        file_name = file_name.strip()
        if not os.path.exists(file_name):
            raise FileNotFoundError(file_name)
        _, ext = os.path.splitext(file_name)
        codec = Snapshotter.READ_CODECS[ext]
        with codec(file_name) as fin:
            return pickle.load(fin)
