# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 5, 2013

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
from veles.mapped_object_registry import MappedObjectsRegistry


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
    hide_from_registry = True is applied to base units which must not be used
    directly, only subclassed. There is also a 'hide_from_registry_all'
    attribute, do disable the registration of the whole inheritance tree, so
    that all the children are automatically hidden.
    """
    units = set()
    hidden_units = set()
    enabled = True

    def __init__(cls, name, bases, clsdict):
        if not UnitRegistry.enabled:
            super(UnitRegistry, cls).__init__(name, bases, clsdict)
            return
        yours = set(cls.mro())
        mine = set(Distributable.mro())
        left = yours - mine
        if len(left) > 1:
            if (not clsdict.get('hide_from_registry', False) and
                    not getattr(cls, 'hide_from_registry_all', False)):
                UnitRegistry.units.add(cls)
            else:
                UnitRegistry.hidden_units.add(cls)
        if "DISABLE_KWARGS_CHECK" in clsdict:
            super(UnitRegistry, cls).__init__(name, bases, clsdict)
            return
        kwattrs = set(getattr(cls, "KWATTRS", set()))
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
                    # 136 = LOAD_DEREF
                    # 106 = LOAD_ATTR
                    # 100 = LOAD_CONST
                    if inst.opcode in (124, 136) and inst.argval == kw_var:
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
                    r"%(kwargs)s\.(get|pop)\(([^\s,\)]+)|%(kwargs)s\[([^\]]+)"
                    % {"kwargs": kw_var})
                src = "".join((l.strip() for l in src)).replace('\n', '')
                for match in kwarg_re.finditer(src):
                    kwattrs.add((match.group(2) or match.group(3))[1:-1])
        cls.KWATTRS = kwattrs
        super(UnitRegistry, cls).__init__(name, bases, clsdict)

    def __call__(cls, *args, **kwargs):
        """ Checks for misprints in argument names """
        obj = super(UnitRegistry, cls).__call__(*args, **kwargs)
        if hasattr(cls, "DISABLE_KWARGS_CHECK") or not UnitRegistry.enabled:
            return obj

        def warning(*largs):
            obj.warning(*largs)
            if root.common.trace.misprints:
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
            ignored_kwargs = set()
            for given_kwarg in set(kwargs).difference(matched):
                candidates = []
                for kwattr in cls.KWATTRS:
                    d = matrix.get((given_kwarg, kwattr))
                    if d == 1:
                        candidates.append(kwattr)
                if len(candidates) == 0:
                    ignored_kwargs.add(given_kwarg)
                else:
                    warning(
                        "Creating %s: potential misprint in keyword argument "
                        "name: expected %s - got %s", obj,
                        " or ".join(candidates), given_kwarg)
            try:
                __IPYTHON__  # pylint: disable=E0602
                from IPython.terminal.interactiveshell import \
                    InteractiveShell
                ignored_kwargs -= set(InteractiveShell.instance().user_ns)
            except NameError:
                pass
            if len(ignored_kwargs) > 0:
                warning(
                    "Creating %s: ignored the following keyword arguments: %s",
                    obj, ", ".join(sorted(ignored_kwargs)))
        return obj


class MappedUnitRegistry(UnitRegistry, MappedObjectsRegistry):
    base = Distributable
