"""
Created on March 12, 2015

Common functions which are used for importing workflow modules.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import sys
from types import ModuleType


def get_file_package_and_module(file_name):
    package_name = os.path.basename(os.path.dirname(file_name))
    module_name = os.path.splitext(os.path.basename(file_name))[0]
    return package_name, module_name


def import_file_as_package(file_name):
    """
    Imports the file as <parent dir>.<file name without ".py">. Raises
    exceptions if fails.
    :param file_name: The path to import.
    :return: The loaded module.
    """
    package_name, module_name = get_file_package_and_module(
        file_name)
    sys.path.insert(0, os.path.dirname(os.path.dirname(file_name)))
    try:
        package = __import__("%s.%s" % (package_name, module_name))
        return getattr(package, module_name)
    finally:
        del sys.path[0]


def import_file_as_module(file_name):
    """
    Imports the file as <file name without ".py">. Raises exceptions if fails.
    :param file_name: The path to import.
    :return: The loaded module.
    """
    _, module_name = get_file_package_and_module(file_name)
    sys.path.insert(0, os.path.dirname(file_name))
    try:
        return __import__(module_name)
    finally:
        del sys.path[0]


def try_to_import_file(file_name):
    """
    Tries to import the file as Python module. First calls
    import_file_as_package() and falls back to import_file_as_module(). If
    fails, keeps silent on any errors and returns the occured exceptions.
    :param file_name: The path to import.
    :return: The loaded module or tuple of length 2 with the exceptions.
    """
    try:
        return import_file_as_package(file_name)
    except Exception as e1:
        try:
            return import_file_as_module(file_name)
        except Exception as e2:
            return e1, e2


def is_module(obj):
    return isinstance(obj, ModuleType)
