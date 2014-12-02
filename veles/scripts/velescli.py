#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
This script starts VELES engine and executes the specified model (workflow).

Contact:
    * g.kuznetsov@samsung.com
    * v.markovtsev@samsung.com

.. argparse::
   :module: veles.scripts.velescli
   :func: create_args_parser_sphinx
   :prog: veles

   ::


"""

import veles

__doc__ += (" " * 7 +  # pylint: disable=W0622
            ("\n" + " " * 7).join(veles.__logo__.split('\n')) +
            u"\u200B\n")

from veles.__main__ import __run__, CommandLineBase


def create_args_parser_sphinx():
    """
    This is a top-level function to please Sphinx.
    """
    return CommandLineBase.init_parser(True)


if __name__ == "__main__":
    __run__()
