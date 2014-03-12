#!/usr/bin/python3.3 -O
"""
Created on Oct 14, 2013

 test Veles for wine.

@author: Seresov Denis <d.seresov@samsung.com>
"""


import argparse
import logging
import pickle
import sys

from veles.config import sconfig


def main():

    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    """
    First version - -config_file - one file with local parameters experiment
    (example wine [orighinal version format])
    """
    parser.add_argument("-config_file", type=str,
                        help="config file ", default='wine_config')
    args = parser.parse_args()
    try:
        print(args.config_file)

        __import__(args.config_file)

        print(" import %s " % (args.config_file))
    except:
        print("not import  %s " % (args.config_file))
        return -1
    w = None
    if sconfig.use_snapshot == 1:
        try:
            fin = open(sconfig.snapshot, "rb")
            """
            sconfig not pickle
            not working
            """
            w = pickle.load(fin)
            fin.close()
        except IOError:
            print("not %s " % (sconfig.snapshot))
            return -2
    else:
        print(" wf -  [%s] " % (sconfig.wf))
        wflib = None
        try:
            wflib = __import__(sconfig.wf)
        except:
            wflib = None
            print("not import wf -  [%s] " % (sconfig.wf))
            return -3
        if not (wflib is None):
            try:
                w = wflib.Workflow()
            except:
                print(" %s not Workflow "
                       % (sconfig.wf))
                return -4
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(" %s.initialize " % (sconfig.wf))
    w.initialize()
    print(" %s.run " % (sconfig.wf))
    w.run()
    w.wait_finish()  # plotters.Graphics().wait_finish()
    logging.debug("End of job")
    return 1

if __name__ == "__main__":
    main()
    sys.exit(0)
