"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 21, 2013

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


import logging
from ctypes import byref, c_short, c_char_p, POINTER

import numpy

import veles.error as error
from veles.loader.file_loader import FileLoaderBase, FileListLoaderBase
from veles.loader.libsndfile import SF_INFO, libsndfile


class SndFileMixin(object):
    @staticmethod
    def open_file(file_name):
        info = SF_INFO()
        info.format = 0
        handle = libsndfile().sf_open(
            c_char_p(file_name.encode()), libsndfile.SFM_READ, byref(info))
        if not handle:
            raise error.BadFormatError(
                "Audio file %s does not exist or is in an unsupported format" %
                file_name)

        if info.channels > 2:
            raise error.BadFormatError("Audio file " + file_name +
                                       " has more than two channels. "
                                       "Only mono or stereo are allowed.")
        return {"handle": handle, "samples": info.frames,
                "sampling_rate": info.samplerate, "channels": info.channels,
                "info": info}

    @staticmethod
    def close_file(opened_data):
        libsndfile().sf_close(opened_data["handle"])

    @staticmethod
    def decode_file(file_name):
        opened_data = SndFileLoaderBase.open_file(file_name)
        handle = opened_data["handle"]
        info = opened_data["info"]
        data = numpy.zeros(info.frames * info.channels, dtype=numpy.int16)
        libsndfile().sf_readf_short(handle,
                                    data.ctypes.data_as(POINTER(c_short)),
                                    info.frames)
        libsndfile().sf_close(handle)
        logging.info("Loaded " + file_name + ": " +
                     info.str_format() + ", " + str(info.frames) +
                     " samples at " + str(info.samplerate) + " Hz with " +
                     str(info.channels) + " channels")
        return {"data": data, "sampling_rate": info.samplerate,
                "samples": info.frames, "channels": info.channels,
                "name": file_name}

    @staticmethod
    def file_format(opened_data):
        return opened_data["info"].str_format()


class SndFileLoaderBase(SndFileMixin, FileLoaderBase):
    """
    Decodes the specified audio file to the raw signed PCM 16 bit format
    using libsndfile.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["file_type"] = "audio"
        kwargs["file_subtypes"] = kwargs.get(
            "file_subtypes", ["basic", "ogg", "wav", "wave", "x-wav",
                              "x-flac"])
        super(SndFileLoaderBase, self).__init__(workflow, **kwargs)


class SndFileListLoaderBase(SndFileMixin, FileListLoaderBase):
    pass
