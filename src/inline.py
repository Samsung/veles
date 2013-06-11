"""
Created on Jun 3, 2013

Runtime compilation of C functions with later call from python.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import os
import time
import error
import sys


class Inline(object):
    """Runtime compilation of C functions with later call from python.

    Attributes:
        sources: C source filenames or pure text.
        function_descriptions: dictionary {function_name: argument_types}
            function_name: string, same as in source_text
            argument_types: string consisting of characters:
                i: int
                f: float
                d: double
                *: numpy array of corresponding type,
                the last character means type of returned value (v: void),
                for example: iif*i means: int function_name(int, int, float*)
        module_name: name of the compiled module.
        module: loaded compiled module.
    """
    def __init__(self):
        self.sources = []
        self.function_descriptions = {}
        self.module_name = ("inline%.3f" % (time.time(), )).replace(".", "_")

    def compile(self):
        s = "#include <Python.h>\n\n"

        for src in self.sources:
            try:
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            except IOError:
                s += src

        x_methods = ""

        s_methods = "\nstatic PyMethodDef x_methods[] = {\n"

        for function_name, argument_types in \
            self.function_descriptions.items():
            s_methods += "    {\"" + function_name + "\", " + \
            "(PyCFunction)x_" + function_name + ", METH_VARARGS, \"" + \
            function_name + ".\"},\n"

            x_method = "\nstatic PyObject *x_" + function_name + \
            "(PyObject *self, PyObject *args)\n" + \
            "{"

            arg_types = []
            arg_names = []
            call_types = []
            formats = []
            for i in range(0, len(argument_types)):
                t = argument_types[i]
                if t == "*":
                    arg_types[len(arg_types) - 1] = "Py_ssize_t"
                    call_types[len(call_types) - 1] += "*"
                    formats[len(formats) - 1] = "n"
                elif t == "i":
                    arg_types.append("int")
                    arg_names.append("arg%d" % (len(arg_types) - 1, ))
                    call_types.append("int")
                    formats.append("i")
                elif t == "f":
                    arg_types.append("float")
                    arg_names.append("arg%d" % (len(arg_types) - 1, ))
                    call_types.append("float")
                    formats.append("f")
                elif t == "d":
                    arg_types.append("double")
                    arg_names.append("arg%d" % (len(arg_types) - 1, ))
                    call_types.append("double")
                    formats.append("d")
                elif t == "v":
                    arg_types.append("void")
                    arg_names.append("arg%d" % (len(arg_types) - 1, ))
                    call_types.append("void")
                    formats.append("v")
                else:
                    raise error.ErrBadFormat("Unknown function argument type.")

            if len(formats) > 1:
                x_method += "\n    " + \
                "\n    ".join((arg_types[i] + " " + arg_names[i] + ";") \
                              for i in range(0, len(arg_types) - 1)) + "\n" + \
                "    if(!PyArg_ParseTuple(args, \"" + \
                "".join(formats[0:-1]) + \
                "\", &" + ", &".join(arg_names[0:-1]) + "))\n" + \
                "        return NULL;\n\n"

            #x_method += "    Py_BEGIN_ALLOW_THREADS\n\n"

            if argument_types[-1] != "v":
                x_method += "    " + arg_types[-1] + \
                " retval = (" + arg_types[-1] + ")"
            else:
                x_method += "    "

            x_method += function_name + "(" + \
            ", ".join(("(" + call_types[i] + ")" + arg_names[i]) \
                      for i in range(0, len(call_types) - 1)) + ");\n"

            #x_method += "\n    Py_END_ALLOW_THREADS\n\n"

            if argument_types[-1] != "v":
                x_method += "    return Py_BuildValue(\"" + \
                formats[-1] + "\", retval);\n}\n"
            else:
                x_method += "    return PyLong_FromLong(0);\n}\n"

            x_methods += x_method

        s_methods += "    {NULL, NULL, 0, NULL}\n" + \
        "};\n\n"

        s += x_methods
        s += s_methods

        s += "static struct PyModuleDef x_module =\n" + \
        "{\n" + \
        "    PyModuleDef_HEAD_INIT,\n" + \
        "    \"" + self.module_name + "\",\n" + \
        "    NULL,\n" + \
        "    -1,\n" + \
        "    x_methods\n" + \
        "};\n\n"

        s += "PyMODINIT_FUNC " + \
        "PyInit_" + self.module_name + "(void)\n" + \
        "{\n" + \
        "    return PyModule_Create(&x_module);\n" + \
        "}\n"

        c_nme = "/tmp/" + self.module_name + ".c"
        so_nme = "/tmp/" + self.module_name + ".so"
        fout = open(c_nme, "w")
        fout.write(s)
        fout.close()

        cmd = "gcc -shared -I/usr/include/python3.3m " + c_nme + \
        " -std=c99 -pedantic -Wall -Werror -fPIC" + \
        " -O3 -march=native -mtune=native -o " + so_nme

        if os.system(cmd):
            raise error.ErrBadFormat("gcc compilation failed.")

        os.unlink(c_nme)

        if "/tmp" not in sys.path:
            sys.path.insert(0, "/tmp")
        self.module = __import__(self.module_name)

        os.unlink(so_nme)

    def execute(self, name, *args):
        args2 = []
        for arg in args:
            if type(arg) == numpy.ndarray:
                args2.append(arg.__array_interface__["data"][0])
            else:
                args2.append(arg)
        return self.module.__getattribute__(name).__call__(*args2)
