#!/usr/bin/python3.3 -O
"""
Created on Oct 15, 2013

 test Veles for wine [2].

@author: Seresov Denis <d.seresov@samsung.com>
"""


import argparse
import logging
import pickle
import sys

from veles.config import sconfig


class veles(object):

    def __init__(self):
        if __debug__:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        parser = argparse.ArgumentParser()
        """
        First version - -config_file - one file with
        local parameters experiment
        (example wine [orighinal version format])
        """
        parser.add_argument("-config_file", type=str,
                        help="config file ", default='wine_config')

        self.args = parser.parse_args()
        self.w = None

    def initialize(self):

        try:
            print(self.args.config_file)
            __import__(self.args.config_file)
            print(" import %s " % (self.args.config_file))
        except:
                print("not import  %s " % (self.args.config_file))
                return -1

        if sconfig.use_snapshot == 1:
            try:
                fin = open(sconfig.snapshot, "rb")
                """
                sconfig not pickle
                not working
                """
                self.w = pickle.load(fin)
                fin.close()
            except IOError:
                    print("not %s " % (sconfig.snapshot))
                    return -2
        else:
            print(" wf -  [%s] " % (sconfig.wf))
            print(sys.path)
            wflib = None
            try:
                wflib = __import__(sconfig.wf)
            except:
                wflib = None
                print("not import wf -  [%s] " % (sconfig.wf))
                return -3
            if not (wflib is None):
                try:
                    self.w = wflib.Workflow()
                except:
                    print(" %s not Workflow " % (sconfig.wf))
                return -4

        print(" %s.initialize " % (sconfig.wf))
        self.w.initialize()

    def run(self):
        print(" %s.run " % (sconfig.wf))
        if not (self.w is None):
            self.w.run()
            self.save_pikcle()
            self.save_result()
            self.w.wait_finish()  # plotters.Graphics().wait_finish()
        logging.debug("End of job")
        return 1

    def save_pikcle(self):
        pass

    def save_rezults(self):
        pass

    def load_pikcle(self):
        pass

    def pause_breakepoint(self):
        pass

    def run_breakpoint(self):
        pass
