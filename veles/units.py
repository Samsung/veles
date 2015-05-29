# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 12, 2013

Classes related to units.

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

import threading
import uuid
import weakref
import sys

import six
from zope.interface import Interface, implementer

from veles.cmdline import CommandLineArgumentsRegistry
from veles.compat import from_none
from veles.config import root, get, validate_kwargs
from veles.distributable import Distributable, TriviallyDistributable, \
    IDistributable
from veles.external.progressbar import spin
from veles.mutable import Bool, LinkableAttribute
from veles.prng.random_generator import RandomGenerator
import veles.thread_pool as thread_pool
from veles.timeit2 import timeit
from veles.unit_registry import UnitRegistry
from veles.verified import Verified


class IUnit(Interface):
    """Unit interface which must be implemented by inherited classes.
    """

    def initialize(**kwargs):
        """Performs the object initialization before execution of the workflow.
        E.g., allocate buffers here. Returns True if you need it to be called
        again, after other units are initialized; otherwise, None or False.

        initialize() is invoked in the same order as run(), including
        open_gate() and effects of gate_block and gate_skip.

        self.is_initialized flag is automatically set after it was executed
        and None or False returned.
        """

    def run():
        """Do the job here.
        """


class UnitCommandLineArgumentsRegistry(UnitRegistry,
                                       CommandLineArgumentsRegistry):
    """
    Enables the usage of CommandLineArgumentsRegistry with classes derived from
    Unit.
    """
    pass


def nothing(*args, **kwargs):
    return {}


class UnitException(Exception):
    def __init__(self, unit, *args, **kwargs):
        super(UnitException, self).__init__(*args, **kwargs)
        self.unit = unit


class NotInitializedError(UnitException):
    pass


class RunAfterStopError(UnitException):
    pass


@six.add_metaclass(UnitRegistry)
class Unit(Distributable, Verified):
    hide_from_registry = True
    """General unit in data stream model.

    Attributes:
        _links_from: dictionary of units it depends on.
        _links_to: dictionary of dependent units.
        _pool: the unique ThreadPool instance.
        _pool_lock_: the lock for getting/setting _pool.
        timers: performance timers for run().
        _gate_block: if evaluates to True, open_gate() and run() are not
                     executed and notification is not sent farther.
        _gate_skip: if evaluates to True, open_gate() and run() will are
                    executed, but notification is not sent farther.
    """

    _pool_ = None
    _pool_lock_ = threading.Lock()
    timers = {}
    visible = True

    def __init__(self, workflow, **kwargs):
        self.name = kwargs.get("name")
        self.view_group = kwargs.get("view_group")
        self._demanded = []
        self._id = str(uuid.uuid4())
        self._links_from = {}
        self._links_to = {}
        super(Unit, self).__init__(**kwargs)
        validate_kwargs(self, **kwargs)
        self.verify_interface(IUnit)
        self._gate_block = Bool(False)
        self._gate_skip = Bool(False)
        self._ignores_gate = Bool(kwargs.get("ignore_gate", False))
        self._run_calls = 0
        self._stopped = False
        self._remembers_gates = True
        timings = get(root.common.timings, None)
        if timings is not None and isinstance(timings, set):
            timings = self.__class__.__name__ in timings
        else:
            timings = False
        self._timings = kwargs.get("timings", timings)
        assert isinstance(self._timings, bool)
        self.workflow = workflow
        self.add_method_to_storage("initialize")
        self.add_method_to_storage("run")
        self.add_method_to_storage("stop")
        if hasattr(self, "generate_data_for_master"):
            self.add_method_to_storage("generate_data_for_master")
        if hasattr(self, "apply_data_from_master"):
            self.add_method_to_storage("apply_data_from_master")
        if hasattr(self, "generate_data_for_slave"):
            self.add_method_to_storage("generate_data_for_slave")
        if hasattr(self, "apply_data_from_slave"):
            self.add_method_to_storage("apply_data_from_slave")

    def init_unpickled(self):
        def wrap_to_measure_time(name):
            func = getattr(self, name, None)
            if func is not None:
                setattr(self, name, self._measure_time(func, Unit.timers))

        # Important: these 4 decorator applications must stand before
        # super(...).init_unpickled since it will call
        # Distributable.init_unpickled which finally makes them thread safe.
        wrap_to_measure_time("generate_data_for_slave")
        wrap_to_measure_time("generate_data_for_master")
        wrap_to_measure_time("apply_data_from_slave")
        wrap_to_measure_time("apply_data_from_master")
        super(Unit, self).init_unpickled()
        self._gate_lock_ = threading.Lock()
        self._run_lock_ = threading.Lock()
        self._is_initialized = False
        if hasattr(self, "run"):
            self.run = self._check_run_conditions(self.run)
            self.run = self._track_call(self.run, "run_was_called")
            self.run = self._measure_time(self.run, Unit.timers)
        if hasattr(self, "initialize"):
            self.initialize = self._ensure_reproducible_rg(self.initialize)
            self.initialize = self._retry_call(self.initialize,
                                               "_is_initialized")
            self.initialize = self._check_attrs(self.initialize, self.demanded)
        if hasattr(self, "stop"):
            self.stop = self._track_call(self.stop, "_stopped")
        Unit.timers[self.id] = 0
        self._workflow_ = lambda: None
        links = "_links_from", "_links_to"
        for this, other in links, reversed(links):
            for unit, value in getattr(self, this).items():
                if isinstance(unit, weakref.ReferenceType):
                    continue
                partner = getattr(unit, other, None)
                if partner is None:
                    partner = {}
                    setattr(unit, "_recovered%s_" % other, partner)
                if self not in partner:
                    partner[weakref.ref(self)] = value
        for link in links:
            attr = "_recovered%s_" % link
            recovered = getattr(self, attr, None)
            if recovered is not None:
                getattr(self, link).update(recovered)
                delattr(self, attr)

    def __del__(self):
        if self.id in Unit.timers:
            del Unit.timers[self.id]

    if six.PY2:
        def remove_refs_to_self(self):
            """Python 2.7 GC is dumb, see
            https://docs.python.org/2/library/gc.html#gc.garbage
            Units are never collected since they hold cyclic references to self
            """
            self.run = self.initialize = self.apply_data_from_slave = \
                self.apply_data_from_master = self.generate_data_for_slave = \
                self.generate_data_for_master = self.drop_slave = nothing

    def __getstate__(self):
        state = super(Unit, self).__getstate__()
        if self.stripped_pickle:
            state["_links_from"] = {}
            state["_links_to"] = {}
        else:
            for name in "_links_from", "_links_to":
                state[name] = {u: v for u, v in getattr(self, name).items()
                               if not isinstance(u, weakref.ReferenceType)}
        return state

    def __repr__(self):
        if getattr(self, "_name", None) is not None:
            return "%s.%s \"%s\"" % (self.__class__.__module__,
                                     self.__class__.__name__,
                                     self.name)
        else:
            return object.__repr__(self)

    def __lt__(self, other):
        if not isinstance(other, Unit):
            raise TypeError("unorderable types: %s() < %s()" % (
                            self.__class__.__name__, other.__class__.__name__))
        if self.workflow != other.workflow:
            raise ValueError("unorderable instances: different parent")
        if self.name != other.name:
            return self.name < other.name
        # Hard case: names are the same. Compare corr. indices in the workflow
        return self.workflow.index_of(self) < other.workflow.index_of(other)

    @property
    def demanded(self):
        return self._demanded

    def demand(self, *args):
        """
        Adds attributes which must be linked before initialize(), setting each
        to None.
        """
        for attr in args:
            if getattr(self, attr, None) is not None:
                continue
            try:
                setattr(self, attr, None)
            except AttributeError as e:
                self.error("Are you trying to set the value of a property "
                           "without a setter?")
                raise from_none(e)
        self.demanded.extend(args)

    @property
    def links_from(self):
        return self._links_from

    @property
    def links_to(self):
        return self._links_to

    @property
    def links_from_sorted(self):
        return Unit._sorted_links(self.links_from)

    @property
    def links_to_sorted(self):
        return Unit._sorted_links(self.links_to)

    @property
    def gate_block(self):
        return self._gate_block

    @gate_block.setter
    def gate_block(self, value):
        if not isinstance(value, Bool):
            raise TypeError("veles.mutable.Bool type was expected")
        self._gate_block = value

    @property
    def gate_skip(self):
        return self._gate_skip

    @gate_skip.setter
    def gate_skip(self, value):
        if not isinstance(value, Bool):
            raise TypeError("veles.mutable.Bool type was expected")
        self._gate_skip = value

    @property
    def ignores_gate(self):
        return self._ignores_gate

    @ignores_gate.setter
    def ignores_gate(self, value):
        if not isinstance(value, Bool):
            raise TypeError("veles.mutable.Bool type was expected")
        self._ignores_gate = value

    @property
    def workflow(self):
        return self._workflow_()

    @workflow.setter
    def workflow(self, value):
        if value is None:
            raise ValueError("Unit must have a hosting Workflow")
        if not hasattr(value, "add_ref"):
            raise TypeError(
                "Attempted to set %s's workflow to something which is not a "
                "unit and does not look like a unit: %s. The first argument of"
                " any unit's constructor must be a workflow. Use veles.dummy."
                "DummyWorkflow if you want to create a standalone unit." %
                (self, value))
        if self.workflow is not None:
            self.workflow.del_ref(self)
        self._workflow_ = weakref.ref(value)
        value.add_ref(self)

    @property
    def launcher(self):
        workflow = self.workflow
        while not workflow.is_main:
            workflow = workflow.workflow
        return workflow.workflow  # workflow, workflow, workflow...

    @property
    def name(self):
        if getattr(self, "_name", None) is not None:
            return self._name
        return self.__class__.__name__

    @name.setter
    def name(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("Unit name must be a string")
        self._name = value

    @property
    def id(self):
        return getattr(self, "_id", "<unknown>")

    @property
    def view_group(self):
        return self._view_group

    @view_group.setter
    def view_group(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("Unit view group must be a string")
        self._view_group = value

    @property
    def timings(self):
        return self._timings

    @property
    def thread_pool(self):
        with Unit._pool_lock_:
            if Unit._pool_ is None:
                Unit._pool_ = thread_pool.ThreadPool(
                    name="units",
                    minthreads=root.common.engine.thread_pool.minthreads,
                    maxthreads=root.common.engine.thread_pool.maxthreads)
        return Unit._pool_

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def is_master(self):
        return self.workflow.is_master

    @property
    def is_slave(self):
        return self.workflow.is_slave

    @property
    def is_standalone(self):
        return self.workflow.is_standalone

    @property
    def interactive(self):
        return self.workflow.interactive

    @property
    def run_was_called(self):
        return self._run_calls > 0

    @run_was_called.setter
    def run_was_called(self, value):
        if value:
            self._run_calls += 1
            if root.common.trace.run:
                self.debug("Call #%d finished @%s", self._run_calls,
                           threading.current_thread().name)
            if ((self.workflow is None or not self.interactive) and
                    not root.common.disable.spinning_run_progress):
                spin()
        else:
            raise ValueError("You can not reset run_was_called flag.")

    @property
    def stopped(self):
        return self._stopped

    @stopped.setter
    def stopped(self, value):
        self._stopped = value

    @property
    def total_run_time(self):
        return Unit.timers[self.id]

    @property
    def average_run_time(self):
        return self.total_run_time / self._run_calls \
            if self._run_calls > 0 else 0

    @staticmethod
    def reset_thread_pool():
        pool = Unit._pool_
        Unit._pool_ = None
        return pool

    def stop(self):
        """
        If run() blocks, interrupt it here.
        By default, do nothing and consider run() to always finish in time.
        """
        pass

    def initialize_dependent(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for unit in self.dependent_units():
            unit.initialize()

    def run_dependent(self):
        """Invokes run() on dependent units on different threads.
        """
        if self.stopped and not isinstance(self, Container):
            return
        links = self.links_to_sorted
        # We must create a copy of gate_block-s because they can change
        # while the loop is working
        gate_blocks = [bool(dst.gate_block) for dst in links]
        for index, dst in enumerate(links):
            if gate_blocks[index] or dst.gate_block:
                continue
            if root.common.trace.run:
                self.debug("%s -> %s (%d/%d) @%s", self, dst, index + 1,
                           len(links), threading.current_thread().name)
            if len(self.links_to) == 1:
                dst._check_gate_and_run(self)
            else:
                if not self.thread_pool.started:
                    self.thread_pool.start()
                self.thread_pool.callInThread(dst._check_gate_and_run, self)

    def dependent_units(self, with_open_gate=False):
        yield self
        walk = []
        visited = {self}
        for child in sorted(self._iter_links(self.links_to)):
            walk.append((child, self))
        # flatten the dependency tree by doing breadth first search
        while len(walk) > 0:
            node, parent = walk.pop(0)
            if node in visited or (with_open_gate and
                                   not node.open_gate(parent)):
                continue
            yield node
            visited.add(node)
            for child in sorted(self._iter_links(node.links_to)):
                walk.append((child, node))

    def open_gate(self, *args):
        """Called before run() or initialize_dependent().

        Returns:
            True: gate is open, can call run() or initialize().
            False: gate is closed, run() and initialize() are ignored.
        """
        if self.ignores_gate:
            # Gate is always open.
            return True
        with self._gate_lock_:
            if not len(self.links_from):
                return True
            for src in args:
                self._set_links_value(self.links_from, src, True)
            if not all(self.links_from.values()):
                return False
            # reset activation flags
            self._close_gate()
        return True

    def close_gate(self):
        with self._gate_lock_:
            self._close_gate()

    def close_upstream(self):
        for other in self._iter_links(self.links_to):
            self._set_links_value(other.links_from, self, False)
        return self

    def link_from(self, *args):
        """Adds notification link.
        """
        with self._gate_lock_:
            for src in args:
                self.links_from[src] = False
                if self._find_reference_cycle():
                    del self.links_from[src]
                    self.links_from[weakref.ref(src)] = False
                    with src._gate_lock_:
                        src.links_to[self] = False
                else:
                    with src._gate_lock_:
                        src.links_to[weakref.ref(self)] = False
        return self

    def unlink_from(self, *args):
        """Unlinks self from src.
        """
        with self._gate_lock_:
            for src in args:
                with src._gate_lock_:
                    self._del_link(src.links_to, self)
                self._del_link(self.links_from, src)
        return self

    def unlink_all(self):
        """Unlinks self from other units.
        """
        self.unlink_before()
        self.unlink_after()
        return self

    def unlink_before(self):
        """
        Detaches all previous units from this one.
        """
        with self._gate_lock_:
            for src in self._iter_links(self.links_from):
                with src._gate_lock_:
                    self._del_link(src.links_to, self)
            self.links_from.clear()
        return self

    def unlink_after(self):
        """
        Detaches all subsequent units from this one.
        """
        with self._gate_lock_:
            for dst in self._iter_links(self.links_to):
                with dst._gate_lock_:
                    self._del_link(dst.links_from, self)
            self.links_to.clear()
        return self

    def insert_after(self, *chain):
        """
        Inserts a series of units between this one and any subsequent ones.
        """
        with self._gate_lock_:
            links_to = self.links_to
            self.split()
            first = chain[0]
            last = chain[-1]
            first.link_from(self)
            for dst in self._iter_links(links_to):
                with dst._gate_lock_:
                    dst.link_from(last)
        return self

    def link_attrs(self, other, *args, **kwargs):
        """
        Assigns attributes from other to self, respecting whether each is
        mutable or immutable. In the latter case, an attribute link is created.

        Parameters:
            two_way: in case of an attribute link with an immutable field,
                     allows/disables editing it's value in this object.
        """
        two_way = kwargs.get("two_way", False)
        for arg in args:
            if (isinstance(arg, tuple) and len(arg) == 2 and
                    isinstance(arg[0], str) and isinstance(arg[1], str)):
                self._link_attr(other, *arg, two_way=two_way)
            elif isinstance(arg, str):
                self._link_attr(other, arg, arg, two_way)
            else:
                raise TypeError(repr(arg) + " is not a valid attributes pair")
        return self

    def describe(self, file=sys.stdout):
        real_name = self.name if self._name is not None else "<not set>"
        res = "\n\033[1;36mUnit:\033[0m \"%s\"\n" % real_name
        res += "\033[1;36mClass:\033[0m %s.%s\n" % (self.__class__.__module__,
                                                    self.__class__.__name__)
        res += "\033[1;36mIncoming links:\033[0m\n"
        for link in self._iter_links(self.links_from):
            res += "\t%s" % repr(link)
        res += "\n\033[1;36mOutgoing links:\033[0m\n"
        for link in self._iter_links(self.links_to):
            res += "\t%s" % repr(link)
        six.print_(res, file=file)

    @classmethod
    def reload(cls):
        from veles.external.pydev import xreload
        xreload(sys.modules[cls.__module__])

    @staticmethod
    def is_immutable(value):
        return (isinstance(value, tuple) or isinstance(value, int) or
                isinstance(value, float) or isinstance(value, complex) or
                isinstance(value, bool) or isinstance(value, str))

    def _close_gate(self):
        for unit in self._iter_links(self.links_from):
            self._set_links_value(self.links_from, unit, False)
            self._set_links_value(unit.links_to, self, False)

    @staticmethod
    def _set_links_value(container, obj, value):
        if obj in container:
            container[obj] = value
        else:
            ref = weakref.ref(obj)
            if ref in container:
                container[ref] = value

    @staticmethod
    def _iter_links(container):
        for obj in container:
            if isinstance(obj, weakref.ReferenceType):
                yield obj()
            else:
                yield obj

    @staticmethod
    def _sorted_links(links):
        links = list(Unit._iter_links(links))
        links.sort(key=lambda u: u.name)
        return links

    def _find_reference_cycle(self):
        pending = set(self.links_from)
        visited = set()
        while len(pending) > 0:
            item = pending.pop()
            if isinstance(item, weakref.ReferenceType):
                continue
            if item is self:
                return True
            if item in visited:
                continue
            visited.add(item)
            pending.update(item.links_from)
        return False

    @staticmethod
    def _del_link(container, obj):
        if obj in container:
            del container[obj]
        else:
            ref = weakref.ref(obj)
            if ref in container:
                del container[ref]

    def _break_cyclic_refs(self, other):
        if other in self.links_from:
            del self.links_from[other]
            self.links_from[weakref.ref(other)] = False

    def _link_attr(self, other, mine, yours, two_way):
        if isinstance(other, Container) and not hasattr(other, yours):
            setattr(other, yours, False)
        try:
            attr = getattr(other, yours)
        except AttributeError as e:
            self.error("Unable to link %s.%s to %s.%s",
                       other, yours, self, mine)
            raise from_none(e)
        if Unit.is_immutable(attr):
            LinkableAttribute(self, mine, (other, yours), two_way=two_way)
        else:
            setattr(self, mine, attr)

    def _check_gate_and_run(self, src):
        """Check gate state and run if it is open.
        """
        if not self.open_gate(src):  # gate has priority over skip
            return
        if self.thread_pool.failure is not None:
            # something went wrong in the thread pool
            return
        # Optionally skip the execution
        if not self.gate_skip:
            # If previous run has not yet finished, discard notification.
            if not self._run_lock_.acquire(False):
                return
            try:
                if not self._is_initialized:
                    self.error("%s is not initialized", self.name)
                    raise NotInitializedError(
                        self, "%s is not initialized" % self)
                self.run()
            finally:
                self._run_lock_.release()
        self.run_dependent()

    def _measure_time(self, fn, storage):
        def wrapped_measure_time(*args, **kwargs):
            res, delta = timeit(fn, *args, **kwargs)
            if self.id in storage:
                storage[self.id] += delta
            if self.timings:
                self.debug("%s took %.6f sec", fn.__name__, delta)
            return res

        name = getattr(fn, '__name__',
                       getattr(fn, 'func', wrapped_measure_time).__name__)
        wrapped_measure_time.__name__ = name + '_measure_time'
        return wrapped_measure_time

    def _check_run_conditions(self, fn):
        def wrapped_check_run_conditions(*args, **kwargs):
            if not self._is_initialized:
                raise NotInitializedError(self, "%s is not initialized" % self)
            if self.stopped:
                if thread_pool.ThreadPool.interrupted:
                    for unit in self._iter_links(self.links_from):
                        unit.gate_block <<= True
                    return
                if root.common.exceptions.run_after_stop:
                    raise RunAfterStopError(
                        self, "%s's run() was called after stop(). Looks like "
                              "you made an error in setting control flow links"
                              ". Workflow: %s." % (self, self.workflow))
                else:
                    self.warning(
                        "run() was called after stop(). Looks like you made an"
                        " error in setting control flow links. Workflow: %s. "
                        "Set root.common.exceptions.run_after_stop to True to "
                        "raise an exception instead", self.workflow)
                    return
            return fn(*args, **kwargs)

        fnname = getattr(fn, '__name__', getattr(
            fn, 'func', wrapped_check_run_conditions).__name__)
        wrapped_check_run_conditions.__name__ = fnname + '_track_call'
        return wrapped_check_run_conditions

    def _track_call(self, fn, name):
        def wrapped_track_call(*args, **kwargs):
            # The order is important!
            res = fn(*args, **kwargs)
            setattr(self, name, True)
            return res

        fnname = getattr(fn, '__name__',
                         getattr(fn, 'func', wrapped_track_call).__name__)
        wrapped_track_call.__name__ = fnname + '_track_call'
        return wrapped_track_call

    def _ensure_reproducible_rg(self, fn):
        storage_attribute_name = "_saved_rg_states"

        def wrapped_reproducible_rg(*args, **kwargs):
            current_states = {}
            storage = getattr(self, storage_attribute_name, None)
            if storage is None:
                storage = {}
            for key, value in self.__dict__.items():
                if isinstance(value, RandomGenerator):
                    state = storage.get(key)
                    if state is None:
                        storage[key] = value.state
                    else:
                        current_states[key] = value.state
                        value.state = state
            if len(storage) > 0:
                setattr(self, storage_attribute_name, storage)
            res = fn(*args, **kwargs)
            for key, value in current_states.items():
                getattr(self, key).state = value
            return res

        fnname = getattr(fn, '__name__',
                         getattr(fn, 'func', wrapped_reproducible_rg).__name__)
        wrapped_reproducible_rg.__name__ = fnname + '_reproducible_rg'
        return wrapped_reproducible_rg

    def _retry_call(self, fn, name):
        def wrapped_retry_call(*args, **kwargs):
            retry = fn(*args, **kwargs)
            assert retry is None or isinstance(retry, bool)
            if not retry:
                setattr(self, name, True)
            return retry

        fnname = getattr(fn, '__name__',
                         getattr(fn, 'func', wrapped_retry_call).__name__)
        wrapped_retry_call.__name__ = fnname + '_track_call'
        return wrapped_retry_call

    def _check_attrs(self, fn, attrs):
        def wrapped_check_attrs(*args, **kwargs):
            validate_kwargs(self, **kwargs)
            for attr in attrs:
                val = getattr(self, attr, None)
                if val is None:
                    raise AttributeError("Attribute %s of unit %s is not "
                                         "linked" % (attr, repr(self)))
            return fn(*args, **kwargs)

        name = getattr(fn, '__name__',
                       getattr(fn, 'func', wrapped_check_attrs).__name__)
        wrapped_check_attrs.__name__ = name + '_check_attrs'
        return wrapped_check_attrs


@implementer(IUnit, IDistributable)
class TrivialUnit(Unit, TriviallyDistributable):
    def initialize(self, **kwargs):
        pass

    def run(self):
        pass


class Container(Unit):
    hide_from_registry = True
