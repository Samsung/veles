# encoding: utf-8
"""
Created on Aug 6, 2013

Provides the essential workflow classes.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import datetime
import hashlib
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

    def run(self):
        self.workflow.on_workflow_finished()


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
        self._units = []
        self.start_point = StartPoint(self)
        self.end_point = EndPoint(self)
        self.negotiates_on_connect = True

    def init_unpickled(self):
        super(Workflow, self).init_unpickled()
        self.thread_pool.register_on_shutdown(self.stop)
        self._is_running = False
        self._sync_event_ = threading.Event()
        self._sync_event_.set()
        del Unit.timers[self]

    def __repr__(self):
        return super(Workflow, self).__repr__() + \
            " with %d units" % len(self.units)

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
        return self._units if hasattr(self, "_units") else []

    def initialize(self, **kwargs):
        fin_text = "all units are initialized"
        maxlen = max([len(u.name) for u in self.units] + [len(fin_text)])
        progress = ProgressBar(maxval=len(self.units),
                               term_width=min(80,
                                              len(self.units) + 8 + maxlen),
                               widgets=[Percentage(), ' ', Bar(), ' ',
                                        ' ' * maxlen])
        self.info("Initializing units in %s...", self.name)
        progress.start()
        for unit in self.start_point.dependecy_list():
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
        self.is_running = True
        self._run_time_started_ = time.time()
        if not self.is_master:
            self.start_point.run_dependent()
        self._sync_event_.wait()
        if self.is_running and self.run_is_blocking:
            self.thread_pool.shutdown(False)
            raise RuntimeError("Workflow synchronization internal failure")

    def stop(self):
        self.on_workflow_finished(True)

    def on_workflow_finished(self, slave_force=False):
        for unit in self.units:
            unit.stop()
        self._run_time_finished_ = time.time()
        self.is_running = False
        if not self.is_slave or slave_force:
            self.workflow.on_workflow_finished()
        else:
            self._do_job_callback_(self.generate_data_for_master())

    def add_ref(self, unit):
        if unit not in self.units:
            self.units.append(unit)

    def del_ref(self, unit):
        if unit in self.units:
            self.units.remove(unit)

    def generate_data_for_master(self):
        data = []
        self.debug("Generating the update for master...")
        for unit in self.units:
            if not unit.negotiates_on_connect:
                data.append(unit.generate_data_for_master())
            else:
                data.append(None)
        self.debug("Done with generating the update for master")
        return data

    def generate_data_for_slave(self, slave):
        """
        Produces a new job, when a slave asks for it. Run by a master.
        """
        if not self.is_running:
            return None
        data = []
        has_data = True
        for unit in self.units:
            if not unit.negotiates_on_connect:
                has_data &= unit.has_data_for_slave
        if not has_data:
            # Try again later
            return False
        self.debug("Generating a job for slave %s", slave.id)
        for unit in self.units:
            if not unit.negotiates_on_connect:
                data.append(unit.generate_data_for_slave(slave))
            else:
                data.append(None)
        self.debug("Done with generating a job for slave %s", slave.id)
        return data

    def apply_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the job from master")
        for i in range(0, len(data)):
            unit = self.units[i]
            if data[i] is not None and not unit.negotiates_on_connect:
                unit.apply_data_from_master(data[i])
        self.debug("Done with applying the job from master")

    def apply_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the update from slave %s", slave.id)
        for i in range(len(self.units)):
            unit = self.units[i]
            if data[i] is not None and not unit.negotiates_on_connect:
                unit.apply_data_from_slave(data[i], slave)
        self.debug("Done with applying the update from slave %s", slave.id)

    def drop_slave(self, slave):
        for i in range(len(self.units)):
            self.units[i].drop_slave(slave)
        self.warning("Dropped the job from %s", slave.id)

    def do_job(self, data, callback):
        """
        Executes this workflow on the given source data. Run by a slave.
        """
        self.apply_data_from_master(data)
        self._do_job_callback_ = callback
        try:
            self.run()
        except:
            self.exception("Failed to run the workflow")
            self.stop()

    def apply_update(self, data, slave):
        """
        Harness the results of a slave's job. Run by a master.
        """
        if len(data) == 0:
            self.drop_slave(slave)
            return
        self.apply_data_from_slave(data, slave)

    def generate_initial_data_for_master(self):
        data = []
        self.debug("Generating the initial data for master...")
        for unit in self.units:
            if unit.negotiates_on_connect:
                data.append(unit.generate_data_for_master())
        self.debug("Done with generating the initial data for master")
        return data

    def generate_initial_data_for_slave(self, slave):
        data = []
        self.debug("Generating the initial data for slave...")
        for unit in self.units:
            if unit.negotiates_on_connect:
                data.append(unit.generate_data_for_slave(slave))
        self.debug("Done with generating the initial data for slave")
        return data

    def apply_initial_data_from_master(self, data):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the initial data from master")
        for i in range(0, len(data)):
            unit = self.units[i]
            if data[i] is not None and unit.negotiates_on_connect:
                unit.apply_data_from_master(data[i])
        self.debug("Done with applying the initial data from master")

    def apply_initial_data_from_slave(self, data, slave):
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        self.debug("Applying the initial data from slave %s", slave.id)
        for i in range(0, len(data)):
            unit = self.units[i]
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
            node.set("label",
                     '<<b>%s</b><br/><font point-size=\"8\">%s</font>>'
                     % (unit.name, inspect.getfile(unit.__class__)))
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
        desc = g.to_string()
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
        for key, value in Unit.timers.items():
            uid = key.__class__.__name__ if not by_name else key.name()
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
            if not self.is_master and hasattr(self, "_run_time_started_") and \
               hasattr(self, "_run_time_finished_"):
                rtime_all = self._run_time_finished_ - self._run_time_started_
                table = PrettyTable("measured", "real", "η,%")
                table.add_row(datetime.timedelta(seconds=time_all),
                              datetime.timedelta(seconds=rtime_all),
                              int(time_all * 100 / rtime_all))
                self.info("Total run time:\n%s", str(table))

    def checksum(self):
        sha1 = hashlib.sha1()
        model_name = sys.modules[self.__module__].__file__
        try:
            with open(model_name, "r") as f:
                sha1.update(f.read().encode())
        except:
            self.exception("Failed to calculate checksum of %s",
                           model_name)
            raise
        return sha1.hexdigest()
