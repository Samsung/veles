#!/usr/bin/python3
# encoding: utf-8
'''
veles is python script which starts platform and
executes user script (called experiment)

@author:     Gennady Kuznetsov
@copyright:  Copyright 2013 Samsung R&D Institute Russia
@contact:    g.kuznetsov@samsung.com
'''


import argparse
import os
import runpy
import sys


def main():
    """VELES Machine Learning Platform Command Line Interface
    Copyright 2013
    Samsung R&D Institute Russia
    Samsung Advanced Software Group
    All rights reserved.
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
# TODO(g.kuznetsov): shall take args from launcher
#    parser.add_argument('-m', '--mode',
#                        help='workflow submission mode
#                        [default: %(default)s]',
#                        default='along',
#                        choices=['along', 'master', 'slave', 'cluster'])
    parser.add_argument('workflow',
                        help='paths to the Python script with workflow')

    parser.add_argument('config',
                        help='paths to config file')
    #, type=FileType('r')) # opens the file
    parser.add_argument('config_list',
                        help="list of configurations like:"
                        "root.global_alptha=0.006"
                        "root.snapshot_prefix='test_pr' (config overwriten)",
                        nargs='*', metavar="config,")

    args = parser.parse_args()
    fname_workflow = os.path.abspath(args.workflow)
    fname_config = os.path.abspath(args.config)
    config_list = args.config_list
    runpy.run_path(fname_config)

    exec("\n".join(config_list))

    runpy.run_path(fname_workflow, run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
