"""
Created on Mar 18, 2013

Classes for custom exceptions

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


class VelesException(Exception):
    """Base class for Veles exceptions.
    """
    pass


class ErrNotExists(VelesException):
    """Exception, raised when something does not exist.
    """
    pass


class ErrExists(VelesException):
    """Exception, raised when something already exists.
    """
    pass


class ErrNotImplemented(VelesException):
    """Exception, raised when something is not implemented.
    """
    pass


class ErrBadFormat(VelesException):
    """Exception, raised when bad format of data occured somethere.
    """
    pass


class ErrOpenCL(VelesException):
    """Exception, raised when OpenCL error occured.
    """
    pass
