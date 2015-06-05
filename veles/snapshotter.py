# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 16, 2014

Base units which export the workflow they are attached to.

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


import bz2
from datetime import datetime
import gzip
import logging
import os
import pyodbc
from six import BytesIO
import snappy
import time
from zope.interface import implementer, Interface

from veles.compat import lzma, from_none, FileNotFoundError
from veles.config import root
from veles.distributable import IDistributable
from veles.mutable import Bool
from veles.pickle2 import pickle, best_protocol
from veles.units import Unit, IUnit


class ISnapshotter(Interface):
    def export_file():
        """
        Writes the snapshot to the file system.
        """

    def export_db():
        """
        Writes the snapshot to the database.
        """


@implementer(IUnit, IDistributable)
class SnapshotterBase(Unit):
    hide_from_registry = True
    """Base class for various data exporting units.

    Defines:
        file_name - the file name of the last snapshot
        time - the time of the last snapshot

    Must be defined before initialize():
        suffix - the file name suffix where to take snapshots

    Attributes:
        compression - the compression applied to pickles: None or '', snappy,
                      gz, bz2, xz
        compression_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocations number
        time_interval - take no more than one snapshot within this time window
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(SnapshotterBase, self).__init__(workflow, **kwargs)
        self.verify_interface(ISnapshotter)
        self.directory = kwargs.get("directory", root.common.snapshot_dir)
        self._odbc = kwargs.get("odbc")
        if self._odbc is not None:
            self._table = kwargs.get("table", "veles")
        self.prefix = kwargs.get("prefix", "")
        self.compression = kwargs.get("compression", "gz")
        self.compression_level = kwargs.get("compression_level", 6)
        self.interval = kwargs.get("interval", 1)
        self.time_interval = kwargs.get("time_interval", 15)
        self.time = 0
        self._skipped_counter = 0
        self.skip = Bool(False)
        self.file_name = ""
        self.suffix = None

    def init_unpickled(self):
        super(SnapshotterBase, self).init_unpickled()
        self.slaves = {}

    def initialize(self, **kwargs):
        self.time = time.time()
        self.debug("Compression is set to %s", self.compression)
        self.debug("interval = %d", self.interval)
        self.debug("time_interval = %f", self.time_interval)
        if self._odbc is not None:
            self._db_ = pyodbc.connect(self._odbc)
            self._cursor_ = self._db_.cursor()

    def run(self):
        if self.is_slave or root.common.disable.snapshotting:
            return
        self._skipped_counter += 1
        if self._skipped_counter < self.interval or self.skip:
            return
        self._skipped_counter = 0
        delta = time.time() - self.time
        if delta < self.time_interval:
            self.debug("%f < %f, dropped", delta, self.time_interval)
            return
        if self._odbc is not None:
            self.export_db()
        else:
            self.export_file()
        self.time = time.time()

    def stop(self):
        if self._odbc is not None:
            self._db_.close()

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


class SnappyFile(object):
    def __init__(self, file_name_or_obj, file_mode,
                 buffer_size=snappy._CHUNK_MAX):
        if isinstance(file_name_or_obj, str):
            self._file = open(file_name_or_obj, file_mode)
        else:
            self._file = file_name_or_obj
        self.buffer = bytearray(buffer_size)
        self.buffer_pos = 0
        if file_mode == "wb":
            self._compressor = snappy.StreamCompressor()
        else:
            self._decompressor = snappy.StreamDecompressor()

    @property
    def mode(self):
        return self._file.mode

    @property
    def fileobj(self):
        return self._file

    def tell(self):
        return self._file.tell()

    def write(self, data):
        while self.buffer_pos + len(data) > len(self.buffer):
            if self.buffer_pos == 0 and len(data) > len(self.buffer):
                size = (len(data) // len(self.buffer)) * len(self.buffer)
                self._file.write(self._compressor.compress(data[:size]))
                break
            remainder = len(self.buffer) - self.buffer_pos
            self.buffer[self.buffer_pos:] = data[:remainder]
            self._file.write(self._compressor.compress(bytes(self.buffer)))
            data = data[remainder:]
            self.buffer_pos = 0
        self.buffer_pos += len(data)
        self.buffer[self.buffer_pos - len(data):self.buffer_pos] = data

    def read(self):
        return self._decompressor.decompress(self._file.read(len(self.buffer)))

    def flush(self):
        if self.buffer_pos > 0:
            self._file.write(self._compressor.compress(
                bytes(self.buffer[:self.buffer_pos])))
            self.buffer_pos = 0
        last = self._compressor.flush()
        if last is not None:
            self._file.write(last)
        self._file.flush()

    def close(self):
        self.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


@implementer(ISnapshotter)
class Snapshotter(SnapshotterBase):
    """Takes workflow snapshots.

    Defines:
        file_name - the file name of the last snapshot
        time - the time of the last snapshot

    Must be defined before initialize():
        suffix - the file name suffix where to take snapshots

    Attributes:
        compression - the compression applied to pickles: None or '', snappy,
                      gz, bz2, xz
        compression_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocation number
        time_interval - take no more than one snapshot within this time window
    """

    WRITE_CODECS = {
        None: lambda n, l: open(n, "wb"),
        "": lambda n, l: open(n, "wb"),
        "snappy": lambda n, _: SnappyFile(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(n, "wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    WRITE_OBJ_CODECS = {
        None: lambda n, l: n,
        "": lambda n, l: n,
        "snappy": lambda n, _: SnappyFile(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(
            fileobj=n, mode="wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    READ_CODECS = {
        ".pickle": lambda name: open(name, "rb"),
        "snappy": lambda n, _: SnappyFile(n, "rb"),
        ".gz": lambda name: gzip.GzipFile(name, "rb"),
        ".bz2": lambda name: bz2.BZ2File(name, "rb"),
        ".xz": lambda name: lzma.LZMAFile(name, "rb")
    }

    def export_file(self):
        ext = ("." + self.compression) if self.compression else ""
        rel_file_name = "%s_%s.%d.pickle%s" % (
            self.prefix, self.suffix, best_protocol, ext)
        self.file_name = os.path.join(self.directory, rel_file_name)
        self.info("Snapshotting to %s..." % self.file_name)
        with self._open_file() as fout:
            pickle.dump(self.workflow, fout, protocol=best_protocol)
        file_name_link = os.path.join(
            self.directory, "%s_current.%d.pickle%s" % (
                self.prefix, best_protocol, ext))
        # Link creation may fail when several processes do this all at once,
        # so try-except here:
        try:
            os.remove(file_name_link)
        except OSError:
            pass
        try:
            os.symlink(rel_file_name, file_name_link)
        except OSError:
            pass

    def export_db(self):
        key = ".".join((self.prefix, self.suffix, str(best_protocol)))
        fio = BytesIO()
        self.info("Preparing the snapshot...")
        with self._open_fobj(fio) as fout:
            pickle.dump(self.workflow, fout, protocol=best_protocol)
        binary = pyodbc.Binary(fio.getvalue())
        self.info("Executing SQL insert into \"%s\" (%d bytes)...",
                  self._table, len(binary))
        self._cursor_.execute(
            "insert into %s(timestamp, id, log_id, workflow, name, data) "
            "values (?, ?, ?, ?, ?, ?);" % self._table, datetime.now(),
            self.launcher.id, self.launcher.log_id,
            self.launcher.workflow.name, key, binary)
        self._db_.commit()

    def _open_file(self):
        return Snapshotter.WRITE_CODECS[self.compression](
            self.file_name, self.compression_level)

    def _open_fobj(self, fobj):
        return Snapshotter.WRITE_OBJ_CODECS[self.compression](
            fobj, self.compression_level)

    @staticmethod
    def import_(file_name):
        file_name = file_name.strip()
        if not os.path.exists(file_name):
            raise FileNotFoundError(file_name)
        _, ext = os.path.splitext(file_name)
        codec = Snapshotter.READ_CODECS[ext]
        with codec(file_name) as fin:
            try:
                return pickle.load(fin)
            except ImportError as e:
                logging.getLogger(Snapshotter.__name__).error(
                    "Are you trying to import snapshot of a different "
                    "workflow?")
                raise from_none(e)
