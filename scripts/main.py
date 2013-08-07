#!/usr/bin/python3.3 -O
"""
Created on Jun 4, 2013

Console command line interface for Veles platform.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import cmd
import logging
import traceback
import pickle


# Imports for conveniece
import numpy
import matplotlib.pyplot as pp
import matplotlib.cm as cm


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../src" % (this_dir))
add_path("%s/../Znicz" % (this_dir))
add_path("%s/../Znicz/tests/functional" % (this_dir))


import config
import thread_pool
import units


class VelesShell(cmd.Cmd):
    """Simple Veles shell.

    Attributes:
        last_exec: last python code had been executed.
        to_exec: python code to be executed.
    """
    def __init__(self):
        self.intro = "Welcome to Veles shell.\n"
        self.prompt = "(Veles) "
        super(VelesShell, self).__init__()
        self.last_exec = ""
        self.to_exec = ""

    def do_quit(self, arg):
        """Exit the shell.
        """
        print("\nWill now exit.\n")
        return True

    def do_exit(self, arg):
        """Exit the shell.
        """
        return self.do_quit(arg)

    def _exec(self, line):
        try:
            exec(line, globals(), globals())
            self.last_exec = line
        except:
            a, b, c = sys.exc_info()
            traceback.print_exception(a, b, c)

    def default(self, line):
        self.to_exec += line
        if len(line) and line[len(line) - 1] == "\\":
            self.to_exec += "\n"
            return
        # thread_pool.pool.request(self.exec_, self.to_exec)
        self._exec(self.to_exec)
        self.to_exec = ""

    def emptyline(self):
        pass

    def do_EOF(self, arg):
        return self.do_quit(arg)


def snapshot(obj, fnme):
    fout = open(fnme, "wb")
    pickle.dump(obj, fout)
    fout.close()


veles_shell = VelesShell()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    global veles_shell
    while True:
        try:
            veles_shell.cmdloop()
            break
        except KeyboardInterrupt:
            pass
    sys.exit()


if __name__ == "__main__":
    main()
