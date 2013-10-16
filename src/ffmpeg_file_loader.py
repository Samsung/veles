"""
Created on Oct 15, 2013

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import os
from subprocess import check_call, CalledProcessError
from snd_file_loader import SndFileLoader


class FFmpegFileLoader(object):
    """
    Decodes the specified audio file to the raw signed PCM 16 bit format
    using libfaad2.
    """

    def __init__(self):
        super(FFmpegFileLoader, self).__init__()

    @staticmethod
    def wav_file_name(file_name):
        return "/tmp/" + os.path.basename(file_name) + ".wav"

    @staticmethod
    def ffmpeg_decode_file(file_name):
        wav_output = FFmpegFileLoader.wav_file_name(file_name)
        if not os.path.exists(wav_output):
            try:
                check_call(["avconv", "-v", "1", "-i", file_name, wav_output])
            except CalledProcessError as e:
                logging.error("Calling avconv failed: %s", repr(e))
                raise

    @staticmethod
    def open_file(file_name):
        wav_output = FFmpegFileLoader.wav_file_name(file_name)
        FFmpegFileLoader.ffmpeg_decode_file(file_name)
        result = SndFileLoader.open_file(wav_output)
        result["format"] = os.path.splitext(
            os.path.basename(file_name))[1][1:].upper()
        return result

    @staticmethod
    def close_file(opened_data):
        SndFileLoader.close_file(opened_data)

    @staticmethod
    def decode_file(file_name):
        wav_file = FFmpegFileLoader.wav_file_name(file_name)
        FFmpegFileLoader.ffmpeg_decode_file(file_name)
        SndFileLoader.decode_file(wav_file)
        os.remove(wav_file)

    @staticmethod
    def file_format(opened_data):
        return opened_data["format"]

    supported_extensions = ["m4a", "mp3"]
