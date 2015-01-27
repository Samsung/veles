"""
Created on Nov 5, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.distributable import Distributable
import inspect
from pyxdameraulevenshtein import damerau_levenshtein_distance
import re


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
    kwarg_re = re.compile(r"kwargs\.get\(([^\s,\)]+)|kwargs\[([^\]]+)")

    def __init__(cls, name, bases, clsdict):
        yours = set(cls.mro())
        mine = set(Distributable.mro())
        left = yours - mine
        if len(left) > 1 and not name.endswith('Base') and \
                not clsdict.get('hide', False) and \
                not getattr(cls, 'hide_all', False):
            UnitRegistry.units.add(cls)
        super(UnitRegistry, cls).__init__(name, bases, clsdict)

    def __call__(cls, *args, **kwargs):
        """ Checks for misprints in argument names """
        obj = super(UnitRegistry, cls).__call__(*args, **kwargs)
        kwattrs = set()
        for base in cls.__mro__:
            try:
                src, _ = inspect.getsourcelines(base.__init__)
            except TypeError:
                continue
            for line in src:
                # IUnit requires kwargs to be named as "kwargs", so apply
                # a simple regular expression here
                match = UnitRegistry.kwarg_re.search(line)
                if match is not None:
                    kwattrs.add((match.group(1) or match.group(2))[1:-1])
        cls.KWATTRS = kwattrs
        # Build the matrix of differences
        matrix = {}
        matched = set()
        for given_kwarg in kwargs:
            for kwattr in kwattrs:
                if (kwattr, given_kwarg) in matrix:
                    continue
                matrix[(given_kwarg, kwattr)] = d = \
                    damerau_levenshtein_distance(given_kwarg, kwattr)
                if d == 0:
                    # perfect match, stop further comparisons
                    matched.add(given_kwarg)
                    break
        if len(matched) < len(kwargs):
            # Find replacement candidates with distance = 1
            ignored_kwargs = []
            for given_kwarg in set(kwargs).difference(matched):
                candidates = []
                for kwattr in kwattrs:
                    d = matrix.get((given_kwarg, kwattr))
                    if d == 1:
                        candidates.append(kwattr)
                if len(candidates) == 0:
                    ignored_kwargs.append(given_kwarg)
                else:
                    obj.warning(
                        "Creating %s: potential misprint in keyword argument "
                        "name: expected %s - got %s", obj,
                        " or ".join(candidates), given_kwarg)
            if len(ignored_kwargs) > 0:
                obj.warning(
                    "Creating %s: ignored the following keyword arguments: %s",
                    obj, ", ".join(ignored_kwargs))
        return obj
