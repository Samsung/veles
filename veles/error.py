# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 18, 2013

Classes for custom exceptions

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
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
