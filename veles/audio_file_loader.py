"""
Created on Oct 15, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import os
from itertools import repeat, chain
from zope.interface import implementer

from veles.snd_file_loader import SndFileLoader
from veles.ffmpeg_file_loader import FFmpegFileLoader
from veles.units import Unit, IUnit


@implementer(IUnit)
class AudioFileLoader(Unit):
    """
    Decodes the specified audio file to the raw signed PCM 16 bit format
    using the decoder based on the file extension.
    """

    available_backends = dict(chain(
        zip(SndFileLoader.supported_extensions, repeat(SndFileLoader)),
        zip(FFmpegFileLoader.supported_extensions, repeat(FFmpegFileLoader))))

    def __init__(self, workflow, name=None):
        super(AudioFileLoader, self).__init__(workflow=workflow,
                                              name=name, view_group="LOADER")
        self.outputs = []
        self.files_list = []
        self.backends = {}

    def initialize(self, **kwargs):
        for fn in self.files_list:
            ext = os.path.splitext(fn)[1][1:]
            backend = AudioFileLoader.available_backends.get(ext)
            if not backend:
                self.backends[fn] = None
                continue
            self.backends[fn] = backend()
            opened_data = self.backends[fn].open_file(fn)
            self.backends[fn].close_file(opened_data)
            logging.info("Checked " + fn + ": " +
                         self.backends[fn].file_format(opened_data) + ", " +
                         str(opened_data["samples"]) + " samples at " +
                         str(opened_data["sampling_rate"]) + " Hz with " +
                         str(opened_data["channels"]) + " channels")

    def run(self):
        for fn in self.files_list:
            self.outputs.append(self.backends[fn].decode_file(fn))
