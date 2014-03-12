#!/usr/bin/python3.3 -O
"""
Created on Oct 15, 2013

 test Veles for wine [2].

@author: Seresov Denis <d.seresov@samsung.com>
"""


import sys

import veles.samples.veles_lib as veles_lib


def main():
    vel = veles_lib.veles()
    vel.initialize()
    vel.run()
    pass

if __name__ == "__main__":
    main()
    sys.exit(0)
