"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
from ctypes import byref, c_short, c_char_p, POINTER

import veles.error as error
from veles.libsndfile import libsndfile, SF_INFO


class SndFileLoader(object):
    """
    Decodes the specified audio file to the raw signed PCM 16 bit format
    using libsndfile.
    """

    def __init__(self):
        super(SndFileLoader, self).__init__()

    @staticmethod
    def open_file(file_name):
        info = SF_INFO()
        info.format = 0
        handle = libsndfile().sf_open(c_char_p(file_name.encode()),
                                      libsndfile.SFM_READ,
                                      byref(info))
        if not handle:
            raise error.ErrBadFormat("Audio file " + file_name + " does not "
                                     "exist or is in an unsupported format.")

        if info.channels > 2:
            raise error.ErrBadFormat("Audio file " + file_name +
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
        opened_data = SndFileLoader.open_file(file_name)
        handle = opened_data["handle"]
        info = opened_data["info"]
        data = numpy.empty(info.frames * info.channels, dtype=numpy.short)
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

    supported_extensions = ["wav", "flac", "ogg", "au"]
