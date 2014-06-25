# encoding: utf-8
"""
Created on Aug 6, 2013

Provides the essential workflow classes.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from collections import OrderedDict, defaultdict
import datetime
import hashlib
from itertools import chain
import os
import sys
import tempfile
import time
import threading
import inspect
from zope.interface import implementer

from veles.config import root
from veles.distributable import IDistributable
from veles.units import Unit, TrivialUnit, IUnit
from veles.external.prettytable import PrettyTable
from veles.external.progressbar import ProgressBar, Percentage, Bar
import veles.external.pydot as pydot


class Repeater(TrivialUnit):
    """Completes a typical control flow cycle, usually joining the first unit
    with the last one.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "PLUMBING")
        super(Repeater, self).__init__(workflow, **kwargs)

    def open_gate(self, src):
        """Gate is always open.
        """
        return True


class UttermostPoint(TrivialUnit):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        super(UttermostPoint, self).__init__(workflow, **kwargs)


class StartPoint(UttermostPoint):
    """Start point of a workflow execution.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Start")
        super(StartPoint, self).__init__(workflow, **kwargs)


class EndPoint(UttermostPoint):
    """End point with semaphore.

    Attributes:
        sem_: semaphore.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "End")
        super(EndPoint, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(EndPoint, self).init_unpickled()
        # on_workflow_finished() applies to Workflow's run time
        del Unit.timers[self.id]

    def run(self):
        self.workflow.on_workflow_finished()


class MultiMap(OrderedDict, defaultdict):
    def __init__(self, default_factory=list, *items, **kwargs):
        OrderedDict.__init__(self, *items, **kwargs)
        defaultdict.__init__(self, default_factory)


@implementer(IUnit, IDistributable)
class Workflow(Unit):
    """Base class for unit sets which are logically connected and belong to
    the same host.

    Attributes:
        start_point: start point.
        end_point: end point.
        _units: the list of units belonging to this workflow.
        _sync: flag which makes Workflow.run() either blocking or non-blocking.
        _sync_event_: threading.Event enabling synchronous run().
    """
    def __init__(self, workflow, **kwargs):
        self._plotters_are_enabled = kwargs.get(
            "disable_plotters", not root.common.plotters_disabled)
        self._sync = kwargs.get("sync", True)
        super(Workflow, self).__init__(workflow,
                                       generate_data_for_slave_threadsafe=True,
                                       apply_data_from_slave_threadsafe=False,
                                       **kwargs)
        self._units = MultiMap()
        self.start_point = StartPoint(self)
        self.end_point = EndPoint(self)
        self.negotiates_on_connect = True
        self._checksum = None

    def init_unpickled(self):
        super(Workflow, self).init_unpickled()
        self.thread_pool.register_on_shutdown(self.stop)
        self._is_running = False
        self._run_time_started_ = time.time()
        self._sync_event_ = threading.Event()
        self._sync_event_.set()
        self._run_time_ = 0
        self._method_time_ = {"run": 0}
        del Unit.timers[self.id]

    def __repr__(self):
        return super(Workflow, self).__repr__() + \
            " with %d units" % len(self)

    def __getitem__(self, key):
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
        return sum([len(units) for units in self._units.values()])

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

    @property
    def plotters_are_enabled(self):
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
        return self.start_point.dependent_list()

    def initialize(self, **kwargs):
        fin_text = "all units are initialized"
        maxlen = max([len(u.name) for u in self] + [len(fin_text)])
        progress = ProgressBar(maxval=len(self),
                               term_width=min(80,
                                              len(self) + 8 + maxlen),
                               widgets=[Percentage(), ' ', Bar(), ' ',
                                        ' ' * maxlen])
        self.info("Initializing units in %s...", self.name)
        progress.start()
        for unit in self.units_in_dependency_order:
            progress.widgets[-1] = unit.name + ' ' * (maxlen - len(unit.name))
            if not self.is_standalone:
                unit.verify_interface(IDistributable)
            try:
                unit.initialize(**kwargs)
            except:
                self.error("Unit \"%s\" failed to initialize", unit.name)
                raise
            progress.inc()
        progress.widgets[-1] = fin_text + ' ' * (maxlen - len(fin_text))
        progress.finish()

    def run(self):
        """Starts executing the workflow. This function is asynchronous,
        parent's on_workflow_finished() method will be called.
        """
        self._run_time_started_ = time.time()
        self.is_running = True
        if not self.is_master:
            self.start_point.run_dependent()
        self._sync_event_.wait()
        if self.is_running and self.run_is_blocking:
            self.thread_pool.shutdown(False)
            raise RuntimeError("Workflow synchronization internal failure")

    def stop(self):
        self.on_workflow_finished(True)

    def on_workflow_finished(self, slave_force=False):
        if not self.is_running:
            # Break an infinite loop if Workflow belongs to Workflow
            return
        for unit in self:
            unit.stop()
        run_time = time.time() - self._run_time_started_
        self._run_time_ += run_time
        self._method_time_["run"] += run_time
        self.is_running = False
        if not self.is_slave or slave_force:
            self.workflow.on_workflow_finished()
        else:
            self._do_job_callback_(self.generate_data_for_master())

    def add_ref(self, unit):
        if unit is self:
            raise ValueError("Attempted to add self to self")
        self._units[unit.name].append(unit)

    def del_ref(self, unit):
        if unit.name in self._units.keys():
            self._units[unit.name].remove(unit)

    def run_timed(fn):
        def wrapped(self, *args, **kwargs):
            t = time.time()
            res = fn(self, *args, **kwargs)
            if self.is_slave:
                self._run_time_ += time.time() - t
            return res
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_run_timed'
        return wrapped

    def method_timed(fn):
        def wrapped(self, *args, **kwargs):
            t = time.time()
            res = fn(self, *args, **kwargs)
            mt = self._method_time_.get(fn.__name__)
            if mt is None:
                mt = 0
            mt += time.time() - t
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
        for unit in self:
            if not unit.negotiates_on_connect:
                data.append(unit.generate_data_for_master())
            else:
                data.append(None)
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
            return False
        self.debug("Generating a job for slave %s", slave.id)
        for unit in self:
            if not unit.negotiates_on_connect:
                data.append(unit.generate_data_for_slave(slave))
            else:
                data.append(None)
        self.debug("Done with generating a job for slave %s", slave.id)
        return data

    @run_timed
    @method_timed
    def apply_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the job from master")
        for i in range(len(data)):
            unit = self[i]
            if data[i] is not None and not unit.negotiates_on_connect:
                unit.apply_data_from_master(data[i])
        self.debug("Done with applying the job from master")

    @run_timed
    @method_timed
    def apply_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        sid = slave.id if slave is not None else "self"
        self.debug("Applying the update from slave %s", sid)
        for i in range(len(self)):
            unit = self[i]
            if data[i] is not None and not unit.negotiates_on_connect:
                unit.apply_data_from_slave(data[i], slave)
        self.debug("Done with applying the update from slave %s", sid)
        return True

    @run_timed
    @method_timed
    def drop_slave(self, slave):
        for i in range(len(self)):
            self[i].drop_slave(slave)
        self.warning("Dropped the job from %s", slave.id)

    def do_job(self, data, update, callback):
        """
        Executes this workflow on the given source data. Run by a slave.
        """
        self.apply_data_from_master(data)
        if update is not None:
            self.apply_data_from_slave(update, None)
        self._do_job_callback_ = callback
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

    def generate_graph(self, filename=None, write_on_disk=True):
        g = pydot.Dot(graph_name="Workflow",
                      graph_type="digraph",
                      bgcolor="transparent")
        g.set_prog("circo")
        visited_units = set()
        boilerplate = set([self.start_point])
        while len(boilerplate) > 0:
            unit = boilerplate.pop()
            visited_units.add(unit)
            node = pydot.Node(hex(id(unit)))
            unit_file_name = os.path.relpath(inspect.getfile(unit.__class__),
                                             root.common.veles_dir)
            node.set("label",
                     '<<b>%s</b><br/><font point-size=\"8\">%s</font>>'
                     % (unit.name, unit_file_name))
            node.set("shape", "rect")
            node.add_style("rounded")
            node.add_style("filled")
            color = Workflow.unit_group_colors.get(unit.view_group, "white")
            node.set("fillcolor", color)
            node.set("gradientangle", "90")
            g.add_node(node)
            for link in unit.links_to.keys():
                g.add_edge(pydot.Edge(hex(id(unit)), hex(id(link))))
                if link not in visited_units and link not in boilerplate:
                    boilerplate.add(link)
        if write_on_disk:
            if not filename:
                (_, filename) = tempfile.mkstemp(".png", "workflow_")
            self.info("Saving the workflow graph to %s", filename)
            g.write(filename, format='png')
        desc = g.to_string().strip()
        self.debug("Graphviz workflow scheme:\n" + desc[:-1])
        return desc

    unit_group_colors = {"PLOTTER": "gold",
                         "WORKER": "greenyellow",
                         "LOADER": "cyan",
                         "TRAINER": "coral",
                         "EVALUATOR": "plum",
                         "SERVICE": "lightgrey"}

    def print_stats(self, by_name=False, top_number=5):
        timers = {}
        key_unit_map = {}
        for unit in self:
            key_unit_map[unit.id] = unit
        for key, value in Unit.timers.items():
            unit = key_unit_map.get(key)
            if unit is None:
                continue
            uid = unit.__class__.__name__ if not by_name else unit.name()
            if id not in timers:
                timers[uid] = 0
            timers[uid] += value
        stats = sorted(timers.items(), key=lambda x: x[1], reverse=True)
        time_all = sum(timers.values())
        if time_all > 0:
            table = PrettyTable("#", "%", "time", "unit")
            table.align["unit"] = "l"
            top_time = 0
            for i in range(1, min(top_number, len(stats)) + 1):
                top_time += stats[i - 1][1]
                table.add_row(i, int(stats[i - 1][1] * 100 / time_all),
                              datetime.timedelta(seconds=stats[i - 1][1]),
                              stats[i - 1][0])
            table.add_row("Σ", int(top_time * 100 / time_all),
                          datetime.timedelta(seconds=top_time), "Top 5")
            self.info("Unit run time statistics top:\n%s", str(table))
            table = PrettyTable("units", "real", "η,%")
            table.add_row(datetime.timedelta(seconds=time_all),
                          datetime.timedelta(seconds=self._run_time_),
                          int(time_all * 100 / self._run_time_))
            self.info("Total run time:\n%s", str(table))
            table = PrettyTable("method", "%", "time")
            table.align["method"] = "l"
            time_all = 0
            for k, v in sorted(self._method_time_.items()):
                time_all += v
                table.add_row(k, int(v * 100 / self._run_time_),
                              datetime.timedelta(seconds=v))
            if self.is_slave:
                table.add_row("Σ", int(time_all * 100 / self._run_time_),
                              datetime.timedelta(seconds=time_all))
            self.info("Workflow methods run time:\n%s", str(table))

    def checksum(self):
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
