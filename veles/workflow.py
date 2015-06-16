# encoding: utf-8
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 6, 2013

Provides the essential workflow classes.

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


from collections import OrderedDict, defaultdict
import datetime
import hashlib
import inspect
from itertools import chain
import json
import logging
import os
import six
import sys
import tempfile
import time
import threading
from zope.interface import implementer

from veles.compat import from_none, FileExistsError
from veles.config import root
from veles.distributable import IDistributable
from veles.mutable import LinkableAttribute
from veles.numpy_json_encoder import NumpyJSONEncoder
from veles.result_provider import IResultProvider
from veles.units import Unit, IUnit, Container
from veles.plumbing import StartPoint, EndPoint, Repeater
from veles.external.prettytable import PrettyTable
from veles.external.progressbar import ProgressBar, Percentage, Bar
import veles.external.pydot as pydot
from veles.timeit2 import timeit


class MultiMap(OrderedDict, defaultdict):
    def __init__(self, default_factory=list, *items, **kwargs):
        OrderedDict.__init__(self, *items, **kwargs)
        defaultdict.__init__(self, default_factory)


class NoMoreJobs(Exception):
    pass


@implementer(IUnit, IDistributable)
class Workflow(Container):
    """Base class for unit sets which are logically connected and belong to
    the same host.

    Attributes:
        start_point: start point.
        end_point: end point.
        negotiates_on_connect: True if data must be sent and received during
        the master-slave handshake; otherwise, False.
        _units: the list of units belonging to this Workflow, in
                semi-alphabetical order.
        _sync: flag which makes Workflow.run() either blocking or non-blocking.
        _sync_event_: threading.Event enabling synchronous run().
        _run_time: the total time workflow has been running for.
        _method_time: Workflow's method timings measured by method_timed
                      decorator. Used mainly to profile master-slave.
        fitness: numeric fitness or None (used by genetic optimization).
    """
    hide_from_registry_all = True

    def __init__(self, workflow, **kwargs):
        self._plotters_are_enabled = kwargs.get(
            "enable_plotters", not root.common.disable.plotting)
        self._sync = kwargs.get("sync", True)  # do not move down
        self._units = tuple()
        self._result_file = kwargs.get("result_file")
        super(Workflow, self).__init__(workflow,
                                       generate_data_for_slave_threadsafe=True,
                                       apply_data_from_slave_threadsafe=True,
                                       **kwargs)
        self._context_units = None
        self.start_point = StartPoint(self)
        self.end_point = EndPoint(self)
        self.negotiates_on_connect = True
        self._checksum = None
        self.debug("My checksum is %s", self.checksum)

    def init_unpickled(self):
        super(Workflow, self).init_unpickled()
        # Important! Save the bound method to variable to avoid dead weak refs
        # See http://stackoverflow.com/questions/19443440/weak-reference-to-python-class-method  # nopep8
        self._stop_ = self.stop
        self.thread_pool.register_on_shutdown(self._stop_)
        self._is_running = False
        self._run_time_started_ = time.time()
        self._sync_event_ = threading.Event()
        self._sync_event_.set()
        self._run_time_ = 0
        self._method_time_ = {"run": 0}
        del Unit.timers[self.id]
        units = self._units
        self._units = MultiMap()
        for unit in units:
            unit.workflow = self

    def __del__(self):
        super(Workflow, self).__del__()
        if Unit._pool_ is not None:
            self.thread_pool.unregister_on_shutdown(self._stop_, False)

    def __getstate__(self):
        state = super(Workflow, self).__getstate__()
        # workaround for Python 2.7 MultiMap pickle incompatibility
        state["_units"] = list(self)
        return state

    def __enter__(self):
        self._context_units = []
        return self

    def __exit__(self, _type, value, traceback):
        for unit in self._context_units:
            self.del_ref(unit)
        del self._context_units

    def __repr__(self):
        return super(Workflow, self).__repr__() + \
            " with %d units" % len(self)

    def __getitem__(self, key):
        """Returns the unit by index or by name.
        """
        if isinstance(key, str):
            units = self._units[key]
            if len(units) == 0:
                del self._units[key]
                raise KeyError()
            if len(units) == 1:
                return units[0]
            return units
        if isinstance(key, int):
            observed = 0
            for units in self._units.values():
                if observed + len(units) > key:
                    return units[key - observed]
                observed += len(units)
            raise IndexError()
        raise TypeError("Key must be either a string or an integer.")

    def __iter__(self):
        """Returns the iterator for units belonging to this Workflow.
        """
        class WorkflowIterator(object):
            def __init__(self, workflow):
                super(WorkflowIterator, self).__init__()
                self._name_iter = workflow._units.values().__iter__()
                self._unit_iter = None

            def __next__(self):
                if self._unit_iter is None:
                    self._unit_iter = iter(next(self._name_iter))
                unit = None
                while unit is None:
                    try:
                        unit = next(self._unit_iter)
                    except StopIteration:
                        self._unit_iter = iter(next(self._name_iter))
                return unit

            def next(self):
                return self.__next__()

        return WorkflowIterator(self)

    def __len__(self):
        """Returns the number of units belonging to this Workflow.
        """
        if getattr(self, "_units", None) is None:
            return 0
        return sum([len(units) for units in self._units.values()]) \
            if hasattr(self, "_units") else 0

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, value):
        self._is_running = value
        if self.run_is_blocking:
            if self.is_running:
                self._sync_event_.clear()
            else:
                self._sync_event_.set()

    @property
    def run_is_blocking(self):
        return self._sync

    @run_is_blocking.setter
    def run_is_blocking(self, value):
        self._sync = value

    @Unit.stopped.setter
    def stopped(self, value):
        for unit in self:
            if value:
                unit.stop()
            else:
                unit.stopped = value
        Unit.stopped.fset(self, value)
        self.debug("stopped -> %s", self.stopped)

    @property
    def plotters_are_enabled(self):
        """There exists an ability to disable plotters in the particular
        Workflow instance.
        """
        return self._plotters_are_enabled

    @plotters_are_enabled.setter
    def plotters_are_enabled(self, value):
        self._plotters_are_enabled = value

    @property
    def units(self):
        units = getattr(self, "_units", {})
        return list(chain(*units.values()))

    @property
    def units_in_dependency_order(self):
        return self.start_point.dependent_units()

    @property
    def is_main(self):
        """
        :return:
            True if this workflow is the topmost workflow, that is, not nested;
            otherwise, False.
        """
        return self.workflow.workflow is self

    @property
    def result_file(self):
        return self._result_file

    @result_file.setter
    def result_file(self, value):
        if value is None:
            self._result_file = None
            return
        if not isinstance(value, six.string_types):
            raise TypeError(
                "result_file must be a string (got %s)" % type(value))
        self._result_file = value

    def initialize(self, **kwargs):
        """Initializes all the units belonging to this Workflow, in dependency
        order.
        """
        try:
            snapshot = kwargs["snapshot"]
        except KeyError:
            raise from_none(KeyError(
                "\"snapshot\" (True/False) must be provided in kwargs"))
        units_number = len(self)
        fin_text = "%d units were initialized" % units_number
        maxlen = max([len(u.name) for u in self] + [len(fin_text)])
        if not self.is_standalone:
            self.verify_interface(IDistributable)
        progress = ProgressBar(maxval=units_number,
                               term_width=min(80, len(self) + 8 + maxlen),
                               widgets=[Percentage(), ' ', Bar(), ' ',
                                        ' ' * maxlen], poll=0)
        progress.widgets[0].TIME_SENSITIVE = True
        self.info("Initializing units in %s...", self.name)
        progress.start()
        units_in_dependency_order = list(self.units_in_dependency_order)
        iqueue = list(units_in_dependency_order)
        while len(iqueue) > 0:
            unit = iqueue.pop(0)
            # Early abort in case of KeyboardInterrupt
            if self.thread_pool.joined:
                break
            progress.widgets[-1] = unit.name + ' ' * (maxlen - len(unit.name))
            progress.update()
            if not self.is_standalone:
                unit.verify_interface(IDistributable)
            try:
                partially = unit.initialize(**kwargs)
            except:
                self.error("Unit \"%s\" failed to initialize", unit.name)
                raise
            if partially:
                iqueue.append(unit)
            else:
                if snapshot and not unit._remembers_gates:
                    unit.close_gate()
                    unit.close_upstream()
                progress.inc()
        progress.widgets[-1] = fin_text + ' ' * (maxlen - len(fin_text))
        progress.finish()
        initialized_units_number = len(units_in_dependency_order)
        if initialized_units_number < units_number:
            self.warning("Not all units were initialized (%d left): %s",
                         units_number - initialized_units_number,
                         set(self) - set(units_in_dependency_order))

    def run(self):
        """Starts executing the workflow. This function is synchronous
        if run_is_blocking, otherwise it returns immediately and the
        parent's on_workflow_finished() method will be called.
        """
        for unit in self:
            assert not unit.stopped, "%s is stopped inside %s" % (unit, self)
        self.debug("Started")
        self._run_time_started_ = time.time()
        self.is_running = True
        if not self.is_master:
            self.event("run", "begin")
        if not self.is_master:
            self.start_point.run_dependent()
        if six.PY3:
            self._sync_event_.wait()
        else:
            while not self._sync_event_.wait(1):
                pass

    def stop(self):
        """Manually interrupts the execution, calling stop() on each bound
        unit.
        """
        self.on_workflow_finished()

    def on_workflow_finished(self):
        if not self.is_running:
            # Break an infinite loop if Workflow belongs to Workflow
            return
        self.log(logging.INFO if self.interactive else logging.DEBUG,
                 "Finished")
        self.stopped = True
        run_time = time.time() - self._run_time_started_
        self._run_time_ += run_time
        self._method_time_["run"] += run_time
        self.is_running = False
        if not self.is_master:
            self.event("run", "end")
        if self.result_file is not None:
            self.write_results()
        if self.is_standalone and self.is_main:
            self.workflow.on_workflow_finished()
        elif self.is_slave:
            self._do_job_callback_(self.generate_data_for_master())

    def add_ref(self, unit):
        """Adds a unit to this workflow. Usually, one does not call this method
        directly, but rather during the construction of the unit itself. Each
        unit requires an instance of Workflow in __init__ and add_ref is
        called inside.
        """
        if unit is self:
            raise ValueError("Attempted to add self to self")
        self._units[unit.name].append(unit)
        if self._context_units is not None:
            self._context_units.append(unit)

    def del_ref(self, unit):
        """Removes a unit from this workflow. This is needed for complete unit
        deletion.
        """
        if unit.name in self._units.keys():
            self._units[unit.name].remove(unit)
        if self._context_units is not None and unit in self._context_units:
            self._context_units.remove(unit)

    def index_of(self, unit):
        for index, child in enumerate(self):
            if child == unit:
                return index
        raise IndexError()

    def run_timed(fn):
        """Decorator function to measure the overall run time.
        """
        def wrapped(self, *args, **kwargs):
            res, delta = timeit(fn, self, *args, **kwargs)
            if self.is_slave:
                self._run_time_ += delta
            return res
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_run_timed'
        return wrapped

    def method_timed(fn):
        """Decorator function to profile particular methods.
        """
        def wrapped(self, *args, **kwargs):
            mt = self._method_time_.get(fn.__name__)
            if mt is None:
                mt = 0
            res, dt = timeit(fn, self, *args, **kwargs)
            mt += dt
            self._method_time_[fn.__name__] = mt
            return res
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_method_timed'
        return wrapped

    @run_timed
    @method_timed
    def generate_data_for_master(self):
        data = []
        self.debug("Generating the update for master...")
        self.event("generate_data", "begin")
        for unit in self.units_in_dependency_order:
            if not unit.negotiates_on_connect:
                try:
                    data.append(unit.generate_data_for_master())
                except:
                    self.error("Unit %s failed to generate data for master",
                               unit)
                    raise
            else:
                data.append(None)
        self.event("generate_data", "end")
        self.debug("Done with generating the update for master")
        return data

    @run_timed
    @method_timed
    def generate_data_for_slave(self, slave):
        """
        Produces a new job, when a slave asks for it. Run by a master.
        """
        if not self.is_running:
            return None
        data = []
        has_data = True
        for unit in self:
            if not unit.negotiates_on_connect:
                has_data &= unit.has_data_for_slave
        if not has_data:
            # Try again later
            self.event("generate_data", "single", slave=slave.id,
                       postpone=True)
            return False
        self.debug("Generating a job for slave %s", slave.id)
        self.event("generate_data", "begin", slave=slave.id)
        for unit in self.units_in_dependency_order:
            if not unit.negotiates_on_connect:
                try:
                    data.append(unit.generate_data_for_slave(slave))
                except NoMoreJobs:
                    self.on_workflow_finished()
                    return None
                except:
                    self.error("Unit %s failed to generate data for slave",
                               unit)
                    raise
            else:
                data.append(None)
        self.event("generate_data", "end", slave=slave.id)
        self.debug("Done with generating a job for slave %s", slave.id)
        return data

    @run_timed
    @method_timed
    def apply_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the job from master")
        self.event("apply_data", "begin")
        for i, unit in enumerate(self.units_in_dependency_order):
            if data[i] is not None and not unit.negotiates_on_connect:
                try:
                    unit.apply_data_from_master(data[i])
                except:
                    self.error("Unit %s failed to apply data from master",
                               unit)
                    raise
        self.event("apply_data", "end")
        self.debug("Done with applying the job from master")

    @run_timed
    @method_timed
    def apply_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        sid = slave.id if slave is not None else "self"
        self.debug("Applying the update from slave %s", sid)
        self.event("apply_data", "begin", slave=sid)
        for i, unit in enumerate(self.units_in_dependency_order):
            if data[i] is not None and not unit.negotiates_on_connect:
                try:
                    unit.apply_data_from_slave(data[i], slave)
                except:
                    self.error("Unit %s failed to apply data from slave", unit)
                    raise
        self.event("apply_data", "end", slave=sid)
        self.debug("Done with applying the update from slave %s", sid)
        return True

    @run_timed
    @method_timed
    def drop_slave(self, slave):
        for i in range(len(self)):
            self[i].drop_slave(slave)
        self.event("drop_slave", "single", slave=slave.id)
        self.warning("Dropped the job from %s", slave.id)

    def do_job(self, data, update, callback):
        """
        Executes this workflow on the given source data. Run by a slave.
        Called by Launcher.
        """
        self.apply_data_from_master(data)
        if update is not None:
            self.apply_data_from_slave(update, None)
        self._do_job_callback_ = callback
        self.stopped = False
        try:
            self.run()
        except:
            self.exception("Failed to do the job")
            self.stop()

    run_timed = staticmethod(run_timed)
    method_timed = staticmethod(method_timed)

    def generate_initial_data_for_master(self):
        data = []
        self.debug("Generating the initial data for master...")
        for unit in self:
            if unit.negotiates_on_connect:
                data.append(unit.generate_data_for_master())
        self.debug("Done with generating the initial data for master")
        return data

    def generate_initial_data_for_slave(self, slave):
        data = []
        self.debug("Generating the initial data for slave...")
        for unit in self:
            if unit.negotiates_on_connect:
                data.append(unit.generate_data_for_slave(slave))
        self.debug("Done with generating the initial data for slave")
        return data

    def apply_initial_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the initial data from master")
        for i in range(0, len(data)):
            unit = self[i]
            if data[i] is not None and unit.negotiates_on_connect:
                unit.apply_data_from_master(data[i])
        self.debug("Done with applying the initial data from master")

    def apply_initial_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the initial data from slave %s", slave.id)
        for i in range(0, len(data)):
            unit = self[i]
            if data[i] is not None and unit.negotiates_on_connect:
                unit.apply_data_from_slave(data[i], slave)
        self.debug("Done with applying the initial data from slave %s",
                   slave.id)

    @property
    def computing_power(self):
        """
        Estimates this slave's computing power for initial perfect balancing.
        Run by a slave.
        """
        return 0

    def filter_unit_graph_attrs(self, val):
        return True

    def generate_graph(self, filename=None, write_on_disk=True,
                       with_data_links=False, background="transparent",
                       quiet=False):
        """Produces a Graphviz PNG image of the unit control flow. Returns the
        DOT graph description (string).
        If write_on_disk is False, filename is ignored. If filename is None, a
        temporary file name is taken.
        """
        g = pydot.Dot(graph_name="Workflow",
                      graph_type="digraph",
                      bgcolor=background,
                      mindist=0.5,
                      overlap="false", outputorder="edgesfirst")
        g.set_prog("circo")
        visited_units = set()
        boilerplate = {self.start_point}
        while len(boilerplate) > 0:
            unit = boilerplate.pop()
            visited_units.add(unit)
            node = pydot.Node(hex(id(unit)))
            unit_file_name = os.path.relpath(inspect.getfile(unit.__class__),
                                             root.common.veles_dir)
            if six.PY2 and unit_file_name.endswith('.pyc'):
                unit_file_name = unit_file_name[:-1]
            node.set("label",
                     '<<b><font point-size=\"18\">%s</font></b><br/>'
                     '<font point-size=\"14\">%s</font>>'
                     % (unit.name, unit_file_name))
            node.set("shape", "rect")
            node.add_style("rounded")
            node.add_style("filled")
            color = Workflow.VIEW_GROUP_COLORS.get(unit.view_group, "white")
            node.set("fillcolor", color)
            node.set("gradientangle", "90")
            if isinstance(unit, Repeater):
                g.set("root", hex(id(unit)))
            g.add_node(node)
            for link in self._iter_links(unit.links_to):
                src_id = hex(id(unit))
                dst_id = hex(id(link))
                if unit.view_group == link.view_group and \
                        unit.view_group in self.VIP_GROUPS:
                    # Force units of the same group to be sequential
                    for _ in range(2):
                        g.add_edge(pydot.Edge(
                            src_id, dst_id, color="#ffffff00"))
                g.add_edge(pydot.Edge(src_id, dst_id, penwidth=3, weight=100))
                if link not in visited_units and link not in boilerplate:
                    boilerplate.add(link)
        if with_data_links:
            # Add data links
            # Circo does not allow to ignore certain edges, so we need to save
            # the intermediate result
            (_, dotfile) = tempfile.mkstemp(".dot", "workflow_")
            g.write(dotfile, format='dot')
            g = pydot.graph_from_dot_file(dotfile)
            os.remove(dotfile)
            # Neato without changing the layout
            g.set_prog("neato -n")

            attrs = defaultdict(list)
            refs = []
            for unit in self:
                for key, val in unit.__dict__.items():
                    if key.startswith('__') and hasattr(unit, key[2:]) and \
                       LinkableAttribute.__is_reference__(val):
                        refs.append((unit, key[2:]) + val)
                    if (val is not None and not Unit.is_immutable(val) and
                            key not in Workflow.HIDDEN_UNIT_ATTRS and
                            not key.endswith('_') and
                            self.filter_unit_graph_attrs(val)):
                        if key[0] == '_' and hasattr(unit, key[1:]):
                            key = key[1:]
                        attrs[id(val)].append((unit, key))
            for ref in refs:
                g.add_edge(pydot.Edge(
                    hex(id(ref[0])), hex(id(ref[2])), constraint="false",
                    label=('"%s"' % ref[1]) if ref[1] == ref[3]
                    else '"%s -> %s"' % (ref[1], ref[3]),
                    fontcolor='gray', fontsize="8.0", color='gray'))
            for vals in attrs.values():
                if len(vals) > 1:
                    for val1 in vals:
                        for val2 in vals:
                            if val1[0] == val2[0]:
                                continue
                            label = ('"%s"' % val1[1]) if val1[1] == val2[1] \
                                else '"%s:%s"' % (val1[1], val2[1])
                            g.add_edge(pydot.Edge(
                                hex(id(val1[0])), hex(id(val2[0])), weight=0,
                                label=label, dir="both", color='gray',
                                fontcolor='gray', fontsize="8.0",
                                constraint="false"))
        if write_on_disk:
            if not filename:
                try:
                    os.mkdir(os.path.join(root.common.cache_dir, "plots"),
                             mode=0o775)
                except FileExistsError:
                    pass
                (_, filename) = tempfile.mkstemp(
                    os.path.splitext(filename)[1], "workflow_",
                    dir=os.path.join(root.common.cache_dir, "plots"))
            if not quiet:
                self.debug("Saving the workflow graph to %s", filename)
            try:
                g.write(filename, format=os.path.splitext(filename)[1][1:])
            except pydot.InvocationException as e:
                if "has no position" not in e.value:
                    raise from_none(e)
                error_marker = "Error: node "
                hex_pos = e.value.find(error_marker) + len(error_marker)
                buggy_id = e.value[hex_pos:hex_pos + len(hex(id(self)))]
                buggy_unit = next(u for u in self if hex(id(u)) == buggy_id)
                self.warning("Looks like %s is not properly linked, unable to "
                             "draw the data links.", buggy_unit)
                return self.generate_graph(filename, write_on_disk, False,
                                           background)
            if not quiet:
                self.info("Saved the workflow graph to %s", filename)
        desc = g.to_string().strip()
        if not quiet:
            self.debug("Graphviz workflow scheme:\n%s", desc)
        return desc, filename

    VIEW_GROUP_COLORS = {"PLOTTER": "gold",
                         "WORKER": "greenyellow",
                         "LOADER": "cyan",
                         "TRAINER": "coral",
                         "EVALUATOR": "plum",
                         "SERVICE": "lightgrey"}

    VIP_GROUPS = {"WORKER", "TRAINER"}

    HIDDEN_UNIT_ATTRS = {"_workflow"}

    def get_unit_run_time_stats(self, by_name=False):
        """
        Returns an iterable of tuples of length 2. First element is the unit
        identifier, second is the accumulated run time.
        :param by_name: If True, use unit name as identifier; otherwise,
                        unit class name.
        """
        timers = {}
        key_unit_map = {}
        for unit in self:
            key_unit_map[unit.id] = unit
        for key, value in Unit.timers.items():
            unit = key_unit_map.get(key)
            if unit is None:
                continue
            uid = unit.__class__.__name__ if not by_name else unit.name
            if id not in timers:
                timers[uid] = 0
            timers[uid] += value
        return sorted(timers.items(), key=lambda x: x[1], reverse=True)

    def print_stats(self, by_name=False, top_number=5):
        """Outputs various time statistics gathered with run_timed and
        method_timed.
        """
        stats = self.get_unit_run_time_stats(by_name)
        time_all = sum(s[1] for s in stats)
        if time_all > 0:
            table = PrettyTable("#", "%", "time", "unit")
            table.align["unit"] = "l"
            top_time = 0
            for i in range(1, min(top_number, len(stats)) + 1):
                top_time += stats[i - 1][1]
                table.add_row(i, int(stats[i - 1][1] * 100 / time_all),
                              datetime.timedelta(seconds=stats[i - 1][1]),
                              stats[i - 1][0])
            table.add_row(u"Σ", int(top_time * 100 / time_all),
                          datetime.timedelta(seconds=top_time), "Top 5")
            self.info(u"Unit run time statistics top:\n%s", table)
            table = PrettyTable("units", "real", u"η,%")
            table.add_row(datetime.timedelta(seconds=time_all),
                          datetime.timedelta(seconds=self._run_time_),
                          int(time_all * 100 / (self._run_time_ or 1)))
            self.info(u"Total run time:\n%s", table)
            table = PrettyTable("method", "%", "time")
            table.align["method"] = "l"
            time_all = 0
            for k, v in sorted(self._method_time_.items()):
                if k == "run":
                    continue
                time_all += v
                if self._run_time_ > 0:
                    table.add_row(k, int(v * 100 / self._run_time_),
                                  datetime.timedelta(seconds=v))
            if self.is_slave:
                table.add_row(u"Σ", int(time_all * 100 / self._run_time_),
                              datetime.timedelta(seconds=time_all))
            if time_all > 0:
                self.info(u"Workflow methods run time:\n%s", table)

    def gather_results(self):
        results = {"id": self.launcher.id, "log_id": self.launcher.log_id}
        for unit in self:
            if IResultProvider.providedBy(unit):
                results.update(unit.get_metric_values())
        return results

    def write_results(self, file=None):
        if file is None:
            file = self.result_file
        if isinstance(file, six.string_types):
            fileobj = open(file, "w")
            need_close = True
        else:
            fileobj = file
            need_close = False
        results = self.gather_results()
        try:
            json.dump(results, fileobj, indent=4, sort_keys=True,
                      cls=NumpyJSONEncoder)
        finally:
            if need_close:
                fileobj.close()
        self.info("Successfully wrote %d results to %s", len(results), file)

    @property
    def checksum(self):
        """Returns the cached checksum of file where this workflow is defined.
        """
        if self._checksum is None:
            sha1 = hashlib.sha1()
            model_name = sys.modules[self.__module__].__file__
            try:
                with open(model_name, "rb") as f:
                    sha1.update(f.read())
            except:
                self.exception("Failed to calculate checksum of %s",
                               model_name)
                raise
            self._checksum = sha1.hexdigest()
        return self._checksum
