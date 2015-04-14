# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 21, 2014

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


import zope.interface.verify as verify


def _incompat(required, implemented):
    if (len(implemented['required']) > len(required['required']) and
            not required['kwargs']):
        return 'implementation requires too many arguments'
    if ((len(implemented['positional']) < len(required['positional']))
            and not implemented['varargs']):
        return "implementation doesn't allow enough arguments"
    if required['kwargs'] and not implemented['kwargs']:
        return "implementation doesn't support keyword arguments"
    if required['varargs'] and not implemented['varargs']:
        return "implementation doesn't support variable arguments"

verify._incompat = _incompat
