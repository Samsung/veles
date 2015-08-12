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
from six import BytesIO, add_metaclass
import snappy
import time
from zope.interface import implementer, Interface

from veles.compat import lzma, from_none, FileNotFoundError
from veles.config import root
from veles.distributable import IDistributable
from veles.external.prettytable import PrettyTable
from veles.mapped_object_registry import MappedObjectsRegistry
from veles.mutable import Bool
from veles.pickle2 import pickle, best_protocol
from veles.result_provider import IResultProvider
from veles.unit_registry import UnitRegistry
from veles.units import Unit, IUnit


class ISnapshotter(Interface):
    def export():
        """
        Performs the actual snapshot generation.
        :return: None
        """


class SnapshotterRegistry(UnitRegistry, MappedObjectsRegistry):
    """Metaclass to record Unit descendants. Used for introspection and
    analytical purposes.
    Classes derived from Unit may contain 'hide' attribute which specifies
    whether it should not appear in the list of registered units. Usually
    hide = True is applied to base units which must not be used directly, only
    subclassed. There is also a 'hide_all' attribute, do disable the
    registration of the whole inheritance tree, so that all the children are
    automatically hidden.
    """
    mapping = "snapshotters"


@implementer(IUnit, IDistributable, IResultProvider)
@add_metaclass(SnapshotterRegistry)
class SnapshotterBase(Unit):
    """Base class for various data exporting units.

    Defines:
        destination - the location of the last snapshot (string).
        time - the time of the last snapshot

    Must be defined before initialize():
        suffix - the file name suffix where to take snapshots

    Attributes:
        compression - the compression applied to pickles: None or '', snappy,
                      gz, bz2, xz
        compression_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocations number
        time_interval - take no more than one snapshot within this time window
        skip - If True, run() is skipped but _skipped_counter is incremented.
    """

    hide_from_registry = True
    SIZE_WARNING_THRESHOLD = 200 * 1000 * 1000

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(SnapshotterBase, self).__init__(workflow, **kwargs)
        self.verify_interface(ISnapshotter)
        self.prefix = kwargs.get("prefix", "")
        if "model_index" in root.common.ensemble:
            self.prefix = ("%04d_" % root.common.ensemble.model_index) + \
                self.prefix
        self._destination = ""
        self.compression = kwargs.get("compression", "gz")
        self.compression_level = kwargs.get("compression_level", 6)
        self.interval = kwargs.get("interval", 1)
        self.time_interval = kwargs.get("time_interval", 15)
        self.time = 0
        self._skipped_counter = 0
        self.skip = Bool(False)
        self._warn_about_size = kwargs.get("warn_about_size", True)
        self.demand("suffix")

    def init_unpickled(self):
        super(SnapshotterBase, self).init_unpickled()
        self._slaves = {}

    def __getstate__(self):
        state = super(SnapshotterBase, self).__getstate__()
        state["_warn_about_size"] = False
        return state

    @property
    def destination(self):
        return self._destination

    @property
    def slaves(self):
        return self._slaves

    @property
    def warn_about_size(self):
        return self._warn_about_size

    @warn_about_size.setter
    def warn_about_size(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "warn_about_size must be boolean (got %s)" % type(value))
        self._warn_about_size = value

    def initialize(self, **kwargs):
        self.time = time.time()
        self.debug("Compression is set to %s", self.compression)
        self.debug("interval = %d", self.interval)
        self.debug("time_interval = %f", self.time_interval)

    def run(self):
        if self.is_slave or root.common.disable.snapshotting:
            return
        self._skipped_counter += 1
        if self._skipped_counter < self.interval or self.skip:
            self.debug("%d < %d or %s, dropped", self._skipped_counter,
                       self.interval, ("False", "True")[int(self.skip)])
            return
        self._skipped_counter = 0
        delta = time.time() - self.time
        if delta < self.time_interval:
            self.debug("%f < %f, dropped", delta, self.time_interval)
            return
        self.export()
        self.time = time.time()
        return True

    def stop(self):
        if self._skipped_counter > 0 and not self.skip:
            self._skipped_counter = 0
            self.export()

    def generate_data_for_slave(self, slave):
        self.slaves[slave.id] = 1

    def generate_data_for_master(self):
        return True

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        self._slave_ended(slave)

    def drop_slave(self, slave):
        if slave.id in self.slaves:
            self._slave_ended(slave)

    def get_metric_names(self):
        return {"Snapshot"}

    def get_metric_values(self):
        return {"Snapshot": self.destination}

    def check_snapshot_size(self, size):
        if size > self.SIZE_WARNING_THRESHOLD and self._warn_about_size:
            self._warn_about_size = False
            psizes = []
            try:
                for unit in self.workflow:
                    unit.stripped_pickle = True
                    psize = len(pickle.dumps(unit, protocol=4))
                    psizes.append((psize, unit))
                    unit.stripped_pickle = False
            except:
                self.warning("The snapshot size looks too big: %d bytes", size)
                return
            import gc
            gc.collect()
            psizes.sort(reverse=True)
            pstable = PrettyTable("Unit", "Size")
            pstable.align["Unit"] = "l"
            for size, unit in psizes[:5]:
                pstable.add_row(str(unit), size)
            self.warning(
                "The snapshot size looks too big: %d bytes. Here are top 5 "
                "big units:\n%s", size, pstable)

    def _slave_ended(self, slave):
        if slave is None:
            return
        if slave.id not in self.slaves:
            return
        del self.slaves[slave.id]
        if not (len(self.slaves) or self.gate_skip or self.gate_block):
            self.run()

    @staticmethod
    def _import_fobj(fobj):
        try:
            obj = pickle.load(fobj)
        except ImportError as e:
            logging.getLogger("Snapshotter").error(
                "Are you trying to import snapshot belonging to a different "
                "workflow?")
            raise from_none(e)
        obj._restored_from_snapshot_ = True
        return obj


class SnappyFile(object):
    def __init__(self, file_name_or_obj, file_mode,
                 buffer_size=snappy._CHUNK_MAX):
        if isinstance(file_name_or_obj, str):
            self._file = open(file_name_or_obj, file_mode)
        else:
            self._file = file_name_or_obj
        self.buffer_pos = 0
        if file_mode == "wb":
            self.buffer = bytearray(buffer_size)
            self._compressor = snappy.StreamCompressor()
        else:
            self.buffer = None
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
                data = data[size:]
                continue
            remainder = len(self.buffer) - self.buffer_pos
            self.buffer[self.buffer_pos:] = data[:remainder]
            self._file.write(self._compressor.compress(bytes(self.buffer)))
            data = data[remainder:]
            self.buffer_pos = 0
        self.buffer_pos += len(data)
        self.buffer[self.buffer_pos - len(data):self.buffer_pos] = data

    def read(self, length=None):
        if length is None:
            return self._decompressor.decompress(self._file.read())
        else:
            if self.buffer is not None and \
                    length <= len(self.buffer) - self.buffer_pos:
                result = self.buffer[self.buffer_pos:self.buffer_pos + length]
                self.buffer_pos += length
                if self.buffer_pos == len(self.buffer):
                    self.buffer = None
                return result
            if self.buffer is None:
                result = bytes()
            else:
                result = self.buffer[self.buffer_pos:]
                length -= len(self.buffer) - self.buffer_pos
            while self.buffer is None or length > 0:
                self.buffer = self._file.read(snappy._CHUNK_MAX + 32)
                self.buffer = self._decompressor.decompress(self.buffer)
                result += self.buffer
                length -= len(self.buffer)
            self.buffer_pos = len(self.buffer) + length
            return result[:length]

    def readline(self):
        result = bytes()
        while True:
            if self.buffer is not None:
                i = self.buffer_pos
            else:
                self.buffer = self._file.read(snappy._CHUNK_MAX + 32)
                self.buffer = self._decompressor.decompress(self.buffer)
                if len(self.buffer) == 0:
                    self.buffer = None
                    continue
                i = 0
            while i < len(self.buffer) and self.buffer[i] != b"\n":
                i += 1
            i += 1
            result += self.buffer[self.buffer_pos:i]
            if i < len(self.buffer):
                self.buffer_pos = i
                return result
            self.buffer = None
            self.buffer_pos = 0

    def flush(self):
        if not hasattr(self, "_compressor"):
            return
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
class SnapshotterToFile(SnapshotterBase):
    """Takes workflow snapshots to the file system.
    """
    MAPPING = "file"

    WRITE_CODECS = {
        None: lambda n, l: open(n, "wb"),
        "": lambda n, l: open(n, "wb"),
        "snappy": lambda n, _: SnappyFile(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(n, "wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    READ_CODECS = {
        "pickle": lambda name: open(name, "rb"),
        "snappy": lambda n: SnappyFile(n, "rb"),
        "gz": lambda name: gzip.GzipFile(name, "rb"),
        "bz2": lambda name: bz2.BZ2File(name, "rb"),
        "xz": lambda name: lzma.LZMAFile(name, "rb")
    }

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(SnapshotterToFile, self).__init__(workflow, **kwargs)
        self.directory = kwargs.get("directory", root.common.dirs.snapshots)

    def export(self):
        ext = ("." + self.compression) if self.compression else ""
        rel_file_name = "%s_%s.%d.pickle%s" % (
            self.prefix, self.suffix, best_protocol, ext)
        self._destination = os.path.abspath(os.path.join(
            self.directory, rel_file_name))
        self.info("Snapshotting to %s..." % self.destination)
        with self._open_file() as fout:
            pickle.dump(self.workflow, fout, protocol=best_protocol)
        self.check_snapshot_size(os.path.getsize(self.destination))
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

    @staticmethod
    def import_(file_name):
        file_name = file_name.strip()
        if not os.path.exists(file_name):
            raise FileNotFoundError(file_name)
        _, ext = os.path.splitext(file_name)
        codec = SnapshotterToFile.READ_CODECS[ext[1:]]
        with codec(file_name) as fin:
            return SnapshotterToFile._import_fobj(fin)

    def _open_file(self):
        return SnapshotterToFile.WRITE_CODECS[self.compression](
            self.destination, self.compression_level)


@implementer(ISnapshotter)
class SnapshotterToDB(SnapshotterBase):
    """Takes workflow snapshots to the database via ODBC.
    """
    MAPPING = "odbc"

    WRITE_CODECS = {
        None: lambda n, l: n,
        "": lambda n, l: n,
        "snappy": lambda n, _: SnappyFile(n, "wb"),
        "gz": lambda n, l: gzip.GzipFile(
            fileobj=n, mode="wb", compresslevel=l),
        "bz2": lambda n, l: bz2.BZ2File(n, "wb", compresslevel=l),
        "xz": lambda n, l: lzma.LZMAFile(n, "wb", preset=l)
    }

    READ_CODECS = {
        "pickle": lambda n: n,
        "snappy": lambda n: SnappyFile(n, "rb"),
        "gz": lambda name: gzip.GzipFile(fileobj=name, mode="rb"),
        "bz2": lambda name: bz2.BZ2File(name, "rb"),
        "xz": lambda name: lzma.LZMAFile(name, "rb")
    }

    def __init__(self, workflow, **kwargs):
        super(SnapshotterToDB, self).__init__(workflow, **kwargs)
        self._odbc = kwargs["odbc"]
        self._table = kwargs.get("table", "veles")

    @property
    def odbc(self):
        return self._odbc

    @property
    def table(self):
        return self._table

    def initialize(self, **kwargs):
        super(SnapshotterToDB, self).initialize(**kwargs)
        self._db_ = pyodbc.connect(self.odbc)
        self._cursor_ = self._db_.cursor()

    def stop(self):
        if self.odbc is not None:
            self._db_.close()

    def export(self):
        self._destination = ".".join(
            (self.prefix, self.suffix, str(best_protocol)))
        fio = BytesIO()
        self.info("Preparing the snapshot...")
        with self._open_fobj(fio) as fout:
            pickle.dump(self.workflow, fout, protocol=best_protocol)
        self.check_snapshot_size(len(fio.getvalue()))
        binary = pyodbc.Binary(fio.getvalue())
        self.info("Executing SQL insert into \"%s\"...", self.table)
        now = datetime.now()
        self._cursor_.execute(
            "insert into %s(timestamp, id, log_id, workflow, name, codec, data"
            ") values (?, ?, ?, ?, ?, ?, ?);" % self.table, now,
            self.launcher.id, self.launcher.log_id,
            self.launcher.workflow.name, self.destination, self.compression,
            binary)
        self._db_.commit()
        self.info("Successfully wrote %d bytes as %s @ %s",
                  len(binary), self.destination, now)

    @staticmethod
    def import_(odbc, table, id_, log_id, name=None):
        conn = pyodbc.connect(odbc)
        cursor = conn.cursor()
        query = "select codec, data from %s where id='%s' and log_id='%s'" % (
            table, id_, log_id)
        if name is not None:
            query += " and name = '%s'" % name
        else:
            query += " order by timestamp desc limit 1"
        cursor.execute(query)
        row = cursor.fetchone()
        codec = SnapshotterToDB.READ_CODECS[row.codec]
        with codec(BytesIO(row.data)) as fin:
            return SnapshotterToDB._import_fobj(fin)

    def get_metric_values(self):
        return {"Snapshot": {"odbc": self.odbc,
                             "table": self.table,
                             "name": self.destination}}

    def _open_fobj(self, fobj):
        return SnapshotterToDB.WRITE_CODECS[self.compression](
            fobj, self.compression_level)


@implementer(ISnapshotter)
class Snapshotter(SnapshotterBase):
    def __new__(cls, *args, **kwargs):
        if "odbc" in kwargs:
            return SnapshotterToDB.__new__(cls)
        else:
            return SnapshotterToFile.__new__(cls)

    @staticmethod
    def import_file(file_name):
        return SnapshotterToFile.import_(file_name)

    @staticmethod
    def import_odbc(odbc, table, id_, log_id, name=None):
        return SnapshotterToDB.import_(odbc, table, id_, log_id, name)
