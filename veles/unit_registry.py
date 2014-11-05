"""
Created on Nov 5, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.distributable import Distributable


class UnitRegistry(type):
    """Metaclass to record Unit descendants. Used for introspection and
    analytical purposes.
    Classes derived from Unit may contain 'hide' attribute which specifies
    whether it should not appear in the list of registered units. Usually
    hide = True is applied to base units which must not be used directly, only
    subclassed. There is also a 'hide_all' attribute, do disable the
    registration of the whole inheritance tree, so that all the children are
    automatically hidden.
    """
    units = set()

    def __init__(cls, name, bases, clsdict):
        yours = set(cls.mro())
        mine = set(Distributable.mro())
        left = yours - mine
        if len(left) > 1 and not name.endswith('Base') and \
                not clsdict.get('hide', False) and \
                not getattr(cls, 'hide_all', False):
            UnitRegistry.units.add(cls)
        super(UnitRegistry, cls).__init__(name, bases, clsdict)
