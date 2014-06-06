"""
Created on Mar 18, 2013

Classes for custom exceptions

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


class VelesException(Exception):
    """Base class for Veles exceptions.
    """
    pass


class NotExistsError(VelesException):
    """Raised when something does not exist.
    """
    pass


class AlreadyExistsError(VelesException):
    """Raised when something already exists.
    """
    pass


class BadFormatError(VelesException):
    """Raised when bad format of data occured somethere.
    """
    pass


class OpenCLError(VelesException):
    """Raised when OpenCL error occured.
    """
    pass


class Bug(VelesException):
    """Raised when something goes wrong but it shouldn't.
    """
    pass


class MasterSlaveCommunicationError(VelesException):
    """Raised when master or slaves discovers data inconsistency during the
    communication.
    """
    pass
