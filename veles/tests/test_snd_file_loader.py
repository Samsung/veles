"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import os
import unittest

from veles import __root__
from veles.loader.libsndfile_loader import SndFileMixin


class Test(unittest.TestCase):

    def tearDown(self):
        pass

    def testSndFileLoader(self):
        loader = SndFileMixin()
        data = loader.decode_file(
            os.path.join(__root__, "veles", "tests", "res", "sawyer.flac"))
        logging.info("%d samples at %d Hz" % (data["data"].size,
                                              data["sampling_rate"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testSndFileLoader']
    unittest.main()
