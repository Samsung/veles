# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jan 23, 2015

Defines classes to save and to load an arbitrary Loader's output for 1 epoch.

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
import gzip
from io import SEEK_END
import os

import numpy
from six import BytesIO
import snappy
from zope.interface import implementer

from veles import error
from veles.compat import from_none, lzma
from veles.config import root
from veles.loader.base import Loader, ILoader, CLASS_NAME, TRAIN
from veles.pickle2 import pickle, best_protocol
from veles.snapshotter import SnappyFile
from veles.units import Unit, IUnit


if not hasattr(gzip, "decompress"):
    def decompress(data):
        """Decompress a gzip compressed string in one shot.
        Return the decompressed string.
        """
        with gzip.GzipFile(fileobj=gzip.io.BytesIO(data)) as f:
            return f.read()

    gzip.decompress = decompress


@implementer(IUnit)
class MinibatchesSaver(Unit):
    """Saves data from Loader to pickle file.
    """
    CODECS = {
        "raw": lambda f, _: f,
        "snappy": lambda f, _: SnappyFile(f, "wb"),
        "gz": lambda f, l: gzip.GzipFile(None, fileobj=f, compresslevel=l),
        "bz2": lambda f, l: bz2.BZ2File(f, compresslevel=l),
        "xz": lambda f, l: lzma.LZMAFile(f, preset=l)
    }

    def __init__(self, workflow, **kwargs):
        super(MinibatchesSaver, self).__init__(workflow, **kwargs)
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        self.file_name = kwargs.get(
            "file_name", os.path.join(root.common.cache_dir,
                                      "minibatches.sav"))
        self.compression = kwargs.get("compression", "snappy")
        self.compression_level = kwargs.get("compression_level", 9)
        self.class_chunk_sizes = kwargs.get("class_chunk_sizes", (0, 0, 1))
        self.offset_table = []
        self.demand(
            "minibatch_data", "minibatch_labels", "minibatch_class",
            "class_lengths", "max_minibatch_size", "minibatch_size",
            "shuffle_limit", "has_labels")

    @property
    def effective_class_chunk_sizes(self):
        chunk_sizes = []
        for ci, cs in enumerate(self.class_chunk_sizes):
            if cs == 0:
                cs = self.max_minibatch_size
            elif cs > self.max_minibatch_size:
                raise ValueError(
                    "%s's chunk size may not exceed max minibatch size = %d ("
                    "got %d)" % (CLASS_NAME[ci], self.max_minibatch_size, cs))
            chunk_sizes.append(cs)
        return tuple(chunk_sizes)

    def initialize(self, **kwargs):
        if self.shuffle_limit != 0:
            raise error.VelesException(
                "You must disable shuffling in your loader (set shuffle_limit "
                "to 0)")
        self.file = open(self.file_name, "wb")
        pickle.dump(self.get_header_data(), self.file, protocol=best_protocol)

    def get_header_data(self):
        return self.compression, self.class_lengths, self.max_minibatch_size, \
            self.effective_class_chunk_sizes, \
            self.minibatch_data.shape, self.minibatch_data.dtype, \
            self.minibatch_labels.shape if self.has_labels else None,\
            self.minibatch_labels.dtype if self.has_labels else None

    def prepare_chunk_data(self):
        self.minibatch_data.map_read()
        self.minibatch_labels.map_read()
        arr_data = numpy.zeros(
            (self.effective_class_chunk_sizes[self.minibatch_class],) +
            self.minibatch_data.shape[1:], dtype=self.minibatch_data.dtype)
        if self.has_labels:
            arr_labels = numpy.zeros(
                (self.effective_class_chunk_sizes[self.minibatch_class],) +
                self.minibatch_labels.shape[1:], self.minibatch_labels.dtype)
        else:
            arr_labels = None
        return arr_data, arr_labels

    def fill_chunk_data(self, prepared, interval):
        prepared[0][:] = self.minibatch_data[interval[0]:interval[1]]
        if self.has_labels:
            prepared[1][:] = self.minibatch_labels[interval[0]:interval[1]]

    def run(self):
        prepared = self.prepare_chunk_data()
        chunk_size = self.effective_class_chunk_sizes[self.minibatch_class]
        chunks_number = int(numpy.ceil(self.max_minibatch_size / chunk_size))
        for i in range(chunks_number):
            self.offset_table.append(numpy.uint64(self.file.tell()))
            file = MinibatchesSaver.CODECS[self.compression](
                self.file, self.compression_level)
            self.fill_chunk_data(
                prepared, (i * chunk_size, (i + 1) * chunk_size))
            pickle.dump(prepared, file, protocol=best_protocol)
            file.flush()

    def stop(self):
        if self.file.closed:
            return
        pos = self.file.tell()
        pickle.dump(self.offset_table, self.file, protocol=best_protocol)
        self.debug("Offset table took %d bytes", self.file.tell() - pos)
        self.file.close()


def decompress_snappy(data):
    bio_in = BytesIO(data)
    bio_out = BytesIO()
    snappy.stream_decompress(bio_in, bio_out)
    return bio_out.getvalue()


@implementer(ILoader)
class MinibatchesLoader(Loader):

    CODECS = {
        "raw": lambda b: b,
        "snappy": decompress_snappy,
        "gz": gzip.decompress,
        "bz2": bz2.decompress,
        "xz": lzma.decompress,
    }
    MAPPING = "minibatches_loader"

    def __init__(self, workflow, **kwargs):
        super(MinibatchesLoader, self).__init__(workflow, **kwargs)
        self.file_name = kwargs["file_name"]
        self.file = None
        self.offset_table = []
        self.chunk_numbers = None
        self.mb_chunk_numbers = None
        self.class_chunk_lengths = None
        self.minibatch_data_shape = None
        self.minibatch_data_dtype = None
        self.minibatch_labels_shape = None
        self.minibatch_labels_dtype = None
        self.decompress = None

    def load_data(self):
        self.file = open(self.file_name, "rb")
        (codec, class_lengths, self.old_max_minibatch_size,
         self.class_chunk_lengths,
         self.minibatch_data_shape, self.minibatch_data_dtype,
         self.minibatch_labels_shape, self.minibatch_labels_dtype) = \
            pickle.load(self.file)
        self.class_lengths[:] = class_lengths
        self._has_labels = self.minibatch_labels_shape is not None
        self.decompress = MinibatchesLoader.CODECS[codec]

        self.chunk_numbers = []
        for ci, cl in enumerate(self.class_lengths):
            mb_chunks = int(numpy.ceil(self.old_max_minibatch_size /
                                       self.class_chunk_lengths[ci]))
            mb_count = int(numpy.ceil(cl / self.old_max_minibatch_size))
            self.chunk_numbers.append(mb_chunks * mb_count)

        class BytesMeasurer(object):
            def __init__(self):
                self.size = 0

            def write(self, data):
                self.size += len(data)

        bm = BytesMeasurer()
        fake_table = [numpy.uint64(i) for i in range(sum(self.chunk_numbers))]
        pickle.dump(fake_table, bm, protocol=best_protocol)
        self.file.seek(-bm.size, SEEK_END)
        try:
            self.offset_table = pickle.load(self.file)
        except pickle.UnpicklingError as e:
            self.error("Failed to read the offset table (table offset was %d)",
                       bm.size)
            raise from_none(e)
        for i, offset in enumerate(self.offset_table):
            self.offset_table[i] = int(offset)
        # Virtual end
        self.offset_table.append(self.file.tell() - bm.size)
        self.debug("Offsets: %s", self.offset_table)
        if self.class_lengths[TRAIN] == 0:
            assert self.normalization_type == "none", \
                "You specified \"%s\" normalization but there are no train " \
                "samples to analyze." % self.normalization_type
            self.normalizer.analyze(self.minibatch_data.mem)

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.minibatch_data_shape[1:],
            dtype=self.minibatch_data_dtype))

    def fill_minibatch(self):
        chunks_map = [
            self.get_address(sample) + (i,) for i, sample in
            enumerate(self.minibatch_indices.mem[:self.minibatch_size])]
        chunks_map.sort()
        prev_chunk_number = -1
        chunk = None
        for chunk_number, chunk_offset, index in chunks_map:
            if prev_chunk_number != chunk_number:
                prev_chunk_number = chunk_number
                self.file.seek(self.offset_table[chunk_number])
                buffer = self.file.read(self.offset_table[chunk_number + 1] -
                                        self.offset_table[chunk_number])
                chunk = pickle.loads(self.decompress(buffer))
            mb_data, mb_labels = chunk
            self.minibatch_data[index] = mb_data[chunk_offset]
            if self.has_labels:
                self.minibatch_labels[index] = mb_labels[chunk_offset]

    def get_address(self, index):
        class_index, class_remainder = self.class_index_by_sample_index(index)
        chunk_length = self.class_chunk_lengths[class_index]
        chunk_number = sum(self.chunk_numbers[:class_index])
        class_offset = self.class_lengths[class_index] - class_remainder
        mb_chunks = int(numpy.ceil(self.old_max_minibatch_size / chunk_length))
        mb_ind, mb_off = divmod(class_offset, self.old_max_minibatch_size)
        chunk_number += mb_ind * mb_chunks
        mb_ind, mb_off = divmod(mb_off, chunk_length)
        return chunk_number, mb_off
