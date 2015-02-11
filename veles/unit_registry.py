"""
Created on Nov 5, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

try:
    from dis import get_instructions
    USE_DIS = True
except ImportError:
    import re
    USE_DIS = False
import inspect
from pyxdameraulevenshtein import damerau_levenshtein_distance
from traceback import extract_stack, format_list

from veles.config import root
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
        kwattrs = set()
        for base in cls.__mro__:
            try:
                kw_var = inspect.getargspec(base.__init__).keywords
            except TypeError:
                continue
            if USE_DIS:
                try:
                    instrs = get_instructions(base.__init__)
                except TypeError:
                    continue
                loading_fast_kwargs = False
                for inst in instrs:
                    # https://hg.python.org/cpython/file/b3f0d7f50544/Include/opcode.h  # nopep8
                    # 124 = LOAD_FAST
                    # 106 = LOAD_ATTR
                    # 100 = LOAD_CONST
                    if inst.opcode == 124 and inst.argval == kw_var:
                        loading_fast_kwargs = True
                    elif loading_fast_kwargs and inst.opcode == 106:
                        continue
                    elif loading_fast_kwargs and inst.opcode == 100:
                        kwattrs.add(inst.argval)
                        loading_fast_kwargs = False
                    else:
                        loading_fast_kwargs = False
            else:
                try:
                    src, _ = inspect.getsourcelines(base.__init__)
                except TypeError:
                    continue
                kwarg_re = re.compile(
                    r"%(kwargs)s\.get\(([^\s,\)]+)|%(kwargs)s\[([^\]]+)" %
                    {"kwargs": kw_var})
                src = "".join((l.strip() for l in src)).replace('\n', '')
                for match in kwarg_re.finditer(src):
                    kwattrs.add((match.group(1) or match.group(2))[1:-1])
        cls.KWATTRS = kwattrs
        super(UnitRegistry, cls).__init__(name, bases, clsdict)

    def __call__(cls, *args, **kwargs):
        """ Checks for misprints in argument names """
        obj = super(UnitRegistry, cls).__call__(*args, **kwargs)

        def warning(*largs):
            obj.warning(*largs)
            if root.common.trace_misprints:
                obj.warning("Stack trace:\n%s",
                            "".join(format_list(extract_stack(
                                inspect.currentframe().f_back.f_back))))

        # Build the matrix of differences
        matrix = {}
        matched = set()
        for given_kwarg in kwargs:
            for kwattr in cls.KWATTRS:
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
                for kwattr in cls.KWATTRS:
                    d = matrix.get((given_kwarg, kwattr))
                    if d == 1:
                        candidates.append(kwattr)
                if len(candidates) == 0:
                    ignored_kwargs.append(given_kwarg)
                else:
                    warning(
                        "Creating %s: potential misprint in keyword argument "
                        "name: expected %s - got %s", obj,
                        " or ".join(candidates), given_kwarg)
            if len(ignored_kwargs) > 0:
                warning(
                    "Creating %s: ignored the following keyword arguments: %s",
                    obj, ", ".join(ignored_kwargs))
        return obj
