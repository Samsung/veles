"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import logging
import error
import units
import numpy
from libsndfile import libsndfile, SF_INFO
from ctypes import byref, c_short, c_char_p, POINTER


class SndFileLoader(units.Unit):
    """
    Decodes the specified audio file to the raw signed PCM 16 bit format
    """

    def __init__(self):
        super(SndFileLoader, self).__init__()
        self.outputs = []
        self.files_list = []

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
        if info.channels > 1:
            raise error.ErrBadFormat("Audio file " + file_name +
                                     " has more than one channel. "
                                     "Only mono is allowed.")
        return handle, info

    @staticmethod
    def decode_file(file_name):
        handle, info = SndFileLoader.open_file(file_name)
        data = numpy.empty(info.frames, dtype=numpy.short)
        libsndfile().sf_readf_short(handle,
                                    data.ctypes.data_as(POINTER(c_short)),
                                    info.frames)
        libsndfile().sf_close(handle)
        logging.info("Loaded " + file_name + ": " +
                     info.str_format() + ", " + str(info.frames) +
                     " samples at " + str(info.samplerate) + " Hz")
        return {"data": data, "sampling_rate": info.samplerate,
                "name": file_name}

    def initialize(self):
        for file in self.files_list:
            handle, info = self.open_file(file)
            libsndfile().sf_close(handle)
            logging.info("Checked " + file + ": " +
                         info.str_format() + ", " + str(info.frames) +
                         " samples at " + str(info.samplerate) + " Hz")

    def run(self):
        for file in self.files_list:
            self.outputs.append(self.decode_file(file))
