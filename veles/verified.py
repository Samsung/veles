# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jan 30, 2015

:class:`Verified` definition.

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


from zope.interface.verify import verifyObject, verifyClass

from veles.compat import from_none
from veles.logger import Logger
import veles.zope_verify_fix  # pylint: disable=W0611


class Verified(Logger):
    """
    Base for all classes which follow any :class:`zope.interface.Interface`.
    """
    def verify_interface(self, iface):
        if getattr(type(self), "DISABLE_INTERFACE_VERIFICATION", False):
            return
        if not iface.providedBy(self):
            raise NotImplementedError(
                "Unit %s does not implement %s interface"
                % (repr(self), iface.__name__))
        try:
            verifyObject(iface, self)
        except Exception as e:
            self.error("%s does not pass verifyObject(%s)", self, iface)
            raise from_none(e)
        try:
            verifyClass(iface, self.__class__)
        except Exception as e:
            self.error("%s does not pass verifyClass(%s)",
                       self.__class__, iface)
            raise from_none(e)
