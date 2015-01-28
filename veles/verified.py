"""
Created on Jan 30, 2015

:class:`Verified` definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from zope.interface.verify import verifyObject, verifyClass

from veles.logger import Logger
import veles.zope_verify_fix  # pylint: disable=W0611


class Verified(Logger):
    """
    Base for all classes which follow any :class:`zope.interface.Interface`.
    """
    def verify_interface(self, iface):
        if not iface.providedBy(self):
            raise NotImplementedError(
                "Unit %s does not implement %s interface" % (repr(self),
                                                             iface.__name__))
        try:
            verifyObject(iface, self)
        except:
            self.error("%s does not pass verifyObject(%s)", str(self),
                       str(iface))
            raise
        try:
            verifyClass(iface, self.__class__)
        except:
            self.error("%s does not pass verifyClass(%s)",
                       str(self.__class__), str(iface))
            raise
