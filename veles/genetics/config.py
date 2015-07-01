# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Sep 8, 2014

Helpers for specifying paramters to optimize in config.

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


from zope.interface import implementer

from veles.config import Config
from veles.genetics.core import Chromosome, Population, IChromosome
from veles.units import Unit


class Tuneable(object):
    def __init__(self, default):
        self._path = None
        self._name = None
        self._addr = None
        self.default = default

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Tuneable's path must be a string (got %s)" % type(value))
        self._path = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "Tuneable's name must be a string (got %s)" % type(value))
        self._name = value

    @property
    def full_name(self):
        return "%s.%s" % (self.path, self.name)

    @property
    def addr(self):
        return self._addr

    @addr.setter
    def addr(self, value):
        if not isinstance(value, tuple):
            raise TypeError(
                "Tuneable's addr must be a tuple (got %s)" % type(value))
        if len(value) != 2:
            raise ValueError(
                "Tuneable's addr must be of length = 2 (container, key)")
        # Check that the address is valid
        value[0][value[1]]
        self._addr = value

    def set(self, value):
        self.addr[0][self.addr[1]] = type(self.default)(value)

    def __ilshift__(self, value):
        self.set(value)

    def details(self):
        return "default: " + self.default

    def __str__(self):
        return "%s{%s}" % (type(self), self.details())

    def __repr__(self):
        return "%s: %s" % (self.full_name, str(self))


class Range(Tuneable):
    """Class for a tunable range.
    """
    def __init__(self, default, min_value, max_value):
        super(Range, self).__init__(default)
        self.min_value = min_value
        self.max_value = max_value

    def set(self, value):
        if value < self.min_value or value > self.max_value:
            raise ValueError(
                "[%s] Value is out of range [%s, %s]: %s" %
                (self.full_name, self.min_value, self.max_value, value))
        super(Range, self).set(value)

    def details(self):
        return "[%s, %s] (default: %s)" % (
            self.min_value, self.max_value, self.default)


def process_config(config, class_to_process, callback):
    """Applies callback to Config tree elements with the specified class.

    Parameters:
        config: instance of the Config object.
        class_to_process: class of the elements to which to apply the callback.
        callback: callback function with 3 arguments:
                  path: path in the Config tree (of type str) to this instance.
                  addr: tuple (container, key) pointing to this instance.
                  name: name of the parameter (of type str).
                  value: value of the parameter (of type class_to_process).
                  The return value is applied back.
    """
    _process_config(config.__path__, config.__dict__, class_to_process,
                    callback)


def _process_config(path, items, class_to_process, callback):
    if isinstance(items, dict):
        to_visit = items.items()
    else:
        to_visit = enumerate(items)
    to_process = {}
    for k, v in sorted(to_visit):
        if isinstance(v, Config):
            _process_config(v.__path__, v.__dict__, class_to_process, callback)
        elif isinstance(v, (dict, list, tuple)):
            _process_config("%s.%s" % (path, k), v, class_to_process, callback)
        elif isinstance(v, class_to_process):
            to_process[k] = v
    for k, v in sorted(to_process.items()):
        items[k] = callback(path, (items, k), k, v)


def fix_config(cfgroot):
    """Replaces all Tuneable values in Config tree with its defaults.

    Parameters:
        cfgroot: instance of the Config object.
    """
    return process_config(cfgroot, Tuneable, _fix_tuneable)


def _fix_tuneable(path, addr, name, value):
    return value.default


def print_config(cfgroot):
    for name, cfg in cfgroot.__content__.items():
        if name != "common":
            cfg.print_()


@implementer(IChromosome)
class ConfigChromosome(Chromosome):
    """Chromosome, based on Config tree's Tuneable elements.
    """

    def __init__(self, unit, *args, **kwargs):
        super(ConfigChromosome, self).__init__(*args, **kwargs)
        self.unit = unit
        self.snapshot = None
        self.config = Config("")

    def init_unpickled(self):
        super(ConfigChromosome, self).init_unpickled()
        self.unit = None

    @property
    def unit(self):
        return self._unit_

    @unit.setter
    def unit(self, value):
        if value is None:
            self._unit_ = None
            return
        if not isinstance(value, Unit):
            raise TypeError("unit must be a Unit (got %s)" % type(value))
        self._unit_ = value

    def evaluate(self):
        self.unit.evaluate(self)

    def copy(self):
        unit = self.unit
        self.unit = None
        clone = super(ConfigChromosome, self).copy()
        clone.unit = unit
        self.unit = unit
        return clone


class ConfigPopulation(Population):
    def on_generation_changed(self):
        super(ConfigPopulation, self).on_generation_changed()
        self.info("Best configuration achieved so far:")
        print_config(self[0].config)
