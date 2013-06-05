#!/usr/bin/python3.3
"""
Created on Jun 4, 2013

Console command line interface for Veles platform.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import cmd
import traceback
import _thread


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
add_path("%s/../src" % (this_dir, ))
add_path("%s/../Znicz" % (this_dir, ))


import inline


uni = inline.Inline()
uni.sources.append("#include <unistd.h>\n"
                   "static int Exit(int status) { _exit(status); }")
uni.function_descriptions = {"Exit": "ii"}
uni.compile()


class VelesShell(cmd.Cmd):
    def __init__(self):
        self.intro = "Welcome to Veles shell.\n"
        self.prompt = "(Veles) "
        super(VelesShell, self).__init__()

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
        except:
            a, b, c = sys.exc_info()
            traceback.print_exception(a, b, c)

    def default(self, line):
        #_thread.start_new_thread(self._exec, (line, ))
        self._exec(line)

    def emptyline(self):
        pass


def main():
    sh = VelesShell()
    while True:
        try:
            sh.cmdloop()
            break
        except KeyboardInterrupt:
            pass
    global uni
    uni.execute("Exit", 0)


if __name__ == "__main__":
    main()
