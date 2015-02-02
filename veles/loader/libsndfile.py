'''
Created on Mar 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
'''


from ctypes import cdll, POINTER, c_char_p, c_short, c_int, c_int64, Structure

from veles.compat import from_none
from veles.logger import Logger


class SF_INFO(Structure):
    _fields_ = [("frames", c_int64),
                ("samplerate", c_int),
                ("channels", c_int),
                ("format", c_int),
                ("sections", c_int),
                ("seekable", c_int)]

    formats = {"SF_FORMAT_WAV": "WAV",
               "SF_FORMAT_AIFF": "AIFF",
               "SF_FORMAT_AU": "AU",
               "SF_FORMAT_RAW": "RAW",
               "SF_FORMAT_PAF": "PAF",
               "SF_FORMAT_SVX": "SVX",
               "SF_FORMAT_NIST": "NIST",
               "SF_FORMAT_VOC": "VOC",
               "SF_FORMAT_IRCAM": "IRCAM",
               "SF_FORMAT_W64": "W64",
               "SF_FORMAT_MAT4": "MAT4",
               "SF_FORMAT_MAT5": "MAT5",
               "SF_FORMAT_PVF": "PVF",
               "SF_FORMAT_XI": "XI",
               "SF_FORMAT_HTK": "HTK",
               "SF_FORMAT_SDS": "SDS",
               "SF_FORMAT_AVR": "AVR",
               "SF_FORMAT_WAVEX": "WAVEX",
               "SF_FORMAT_SD2": "SD2",
               "SF_FORMAT_FLAC": "FLAC",
               "SF_FORMAT_CAF": "CAF",
               "SF_FORMAT_WVE": "WVE",
               "SF_FORMAT_OGG": "OGG",
               "SF_FORMAT_MPC2K": "MPC2K",
               "SF_FORMAT_RF64": "RF64"}

    def str_format(self):
        maxval = 0
        maxfmt = ""
        for fmt in SF_INFO.formats:
            val = libsndfile.__dict__[fmt]
            if val & self.format == val and val > maxval:
                maxval = val
                maxfmt = fmt
        return SF_INFO.formats[maxfmt]


class SNDFILE(Structure):
    pass


class libsndfile(Logger):
    '''
    Loads the libsndfile.so shared library and wraps the handle.
    '''

    def __init__(self, path=None):
        if self._handle is not None:
            return
        super(libsndfile, self).__init__()
        self.debug("Initializing a new instance of libsndfile class "
                   "(path is %s)", path)
        if not path:
            self.info("Library path was not specified, "
                      "will use the default (libsndfile.so.1)")
            path = "libsndfile.so.1"
        self._path = path
        try:
            self.debug("Trying to load %s...", path)
            self._handle = cdll.LoadLibrary(path)
        except OSError as e:
            self.critical("Failed to load %s", path)
            raise from_none(e)
        self.debug("Success. Loading functions...")
        self._handle.sf_open.argtypes = [c_char_p, c_int, POINTER(SF_INFO)]
        self._handle.sf_open.restype = POINTER(SNDFILE)
        self._handle.sf_close.argtypes = [POINTER(SNDFILE)]
        self._handle.sf_close.restype = c_int
        self._handle.sf_readf_short.argtypes = [POINTER(SNDFILE),
                                                POINTER(c_short),
                                                c_int64]
        self._handle.sf_readf_short.restype = c_int64
        self.debug("Finished loading functions")

    def __new__(cls, path=None):
        if not cls._instance:
            cls._instance = super(libsndfile, cls).__new__(cls)
            cls._instance._handle = None
        return cls._instance

    def __str__(self):
        return 'libsndfile cdll library handle ' + self._handle

    def __getattr__(self, attr):
        if self._handle is None:
            self.error("Attempted to invoke a function but the library "
                       "was not loaded")
            raise AttributeError()
        return self._handle.__getattribute__(attr)

    _instance = None
    _handle = None
    _path = None
    __all__ = ['sf_open', 'sf_close', 'sf_readf_short']
    SFM_READ = 0x10
    SFM_WRITE = 0x20
    SFM_RDWR = 0x30
    SF_FORMAT_WAV = 0x010000
    SF_FORMAT_AIFF = 0x020000
    SF_FORMAT_AU = 0x030000
    SF_FORMAT_RAW = 0x040000
    SF_FORMAT_PAF = 0x050000
    SF_FORMAT_SVX = 0x060000
    SF_FORMAT_NIST = 0x070000
    SF_FORMAT_VOC = 0x080000
    SF_FORMAT_IRCAM = 0x0A0000
    SF_FORMAT_W64 = 0x0B0000
    SF_FORMAT_MAT4 = 0x0C0000
    SF_FORMAT_MAT5 = 0x0D0000
    SF_FORMAT_PVF = 0x0E0000
    SF_FORMAT_XI = 0x0F0000
    SF_FORMAT_HTK = 0x100000
    SF_FORMAT_SDS = 0x110000
    SF_FORMAT_AVR = 0x120000
    SF_FORMAT_WAVEX = 0x130000
    SF_FORMAT_SD2 = 0x160000
    SF_FORMAT_FLAC = 0x170000
    SF_FORMAT_CAF = 0x180000
    SF_FORMAT_WVE = 0x190000
    SF_FORMAT_OGG = 0x200000
    SF_FORMAT_MPC2K = 0x210000
    SF_FORMAT_RF64 = 0x220000
    SF_FORMAT_PCM_S8 = 0x0001
    SF_FORMAT_PCM_16 = 0x0002
    SF_FORMAT_PCM_24 = 0x0003
    SF_FORMAT_PCM_32 = 0x0004
    SF_FORMAT_PCM_U8 = 0x0005
    SF_FORMAT_FLOAT = 0x0006
    SF_FORMAT_DOUBLE = 0x0007
    SF_FORMAT_ULAW = 0x0010
    SF_FORMAT_ALAW = 0x0011
    SF_FORMAT_IMA_ADPCM = 0x0012
    SF_FORMAT_MS_ADPCM = 0x0013
    SF_FORMAT_GSM610 = 0x0020
    SF_FORMAT_VOX_ADPCM = 0x0021
    SF_FORMAT_G721_32 = 0x0030
    SF_FORMAT_G723_24 = 0x0031
    SF_FORMAT_G723_40 = 0x0032
    SF_FORMAT_DWVW_12 = 0x0040
    SF_FORMAT_DWVW_16 = 0x0041
    SF_FORMAT_DWVW_24 = 0x0042
    SF_FORMAT_DWVW_N = 0x0043
    SF_FORMAT_DPCM_8 = 0x0050
    SF_FORMAT_DPCM_16 = 0x0051
    SF_FORMAT_VORBIS = 0x0060
    SF_ENDIAN_FILE = 0x00000000
    SF_ENDIAN_LITTLE = 0x10000000
    SF_ENDIAN_BIG = 0x20000000
    SF_ENDIAN_CPU = 0x30000000
    SF_FORMAT_SUBMASK = 0x0000FFFF
    SF_FORMAT_TYPEMASK = 0x0FFF0000
    SF_FORMAT_ENDMASK = 0x30000000
