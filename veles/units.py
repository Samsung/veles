"""
Created on Mar 12, 2013

Classes related to units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import six
import threading
import uuid
import weakref
from zope.interface import Interface, implementer

from veles.cmdline import CommandLineArgumentsRegistry
from veles.compat import from_none
from veles.config import root, get, validate_kwargs
from veles.distributable import Distributable, TriviallyDistributable, \
    IDistributable
import veles.error as error
from veles.mutable import Bool, LinkableAttribute
import veles.thread_pool as thread_pool
from veles.timeit import timeit
from veles.unit_registry import UnitRegistry
from veles.verified import Verified


class IUnit(Interface):
    """Unit interface which must be implemented by inherited classes.
    """

    def initialize(**kwargs):
        """Performs the object initialization before execution of the workflow.
        E.g., allocate buffers here.

        initialize() is invoked in the same order as run(), including
        open_gate() and effects of gate_block and gate_skip.

        self.is_initialized flag is automatically set after it was executed.
        """

    def run():
        """Do the job here.
        """

    def stop():
        """If run() blocks, interrupt it here.
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


@six.add_metaclass(UnitRegistry)
class Unit(Distributable, Verified):
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

    class NotInitializedError(Exception):
        pass

    def __init__(self, workflow, **kwargs):
        self.name = kwargs.get("name")
        self.view_group = kwargs.get("view_group")
        self._demanded = []
        self._id = str(uuid.uuid4())
        super(Unit, self).__init__(**kwargs)
        validate_kwargs(self, **kwargs)
        self.verify_interface(IUnit)
        self._links_from = {}
        self._links_to = {}
        self._gate_block = Bool(False)
        self._gate_skip = Bool(False)
        self._run_calls = 0
        timings = get(root.common.timings, None)
        if timings is not None and isinstance(timings, set):
            timings = self.__class__.__name__ in timings
        else:
            timings = False
        self._timings = kwargs.get("timings", timings)
        self._workflow = None
        self.workflow = workflow
        self.add_method_to_storage("initialize")
        self.add_method_to_storage("run")
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
            self.run = self._track_call(self.run, "run_was_called")
            self.run = self._measure_time(self.run, Unit.timers)
        if hasattr(self, "initialize"):
            self.initialize = self._track_call(self.initialize,
                                               "_is_initialized")
            self.initialize = self._check_attrs(self.initialize, self.demanded)
        Unit.timers[self.id] = 0

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
        if isinstance(self.workflow, weakref.ProxyTypes):
            self.warning("Parent workflow %s was discarded during pickling "
                         "because it was weakly referenced (did you call "
                         "detach()?)", self.workflow)
            state["_workflow"] = None
        if self.stripped_pickle:
            state["_links_from"] = {}
            state["_links_to"] = {}
            state["_workflow"] = None
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
            try:
                setattr(self, attr, None)
            except AttributeError:
                self.error("Are you trying to set the value of a property "
                           "without a setter?")
                raise
            self.demanded.append(attr)

    @property
    def links_from(self):
        return self._links_from

    @property
    def links_to(self):
        return self._links_to

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
    def workflow(self):
        return self._workflow

    @workflow.setter
    def workflow(self, value):
        if value is None:
            raise error.VelesException("Unit must have a hosting Workflow")
        if not hasattr(value, "add_ref"):
            raise TypeError(
                "Attempted to set %s's workflow to something which is not a "
                "unit and does not look like a unit: %s. The first argument of"
                " any unit's constructor must be a workflow. Use veles.dummy."
                "DummyWorkflow if you want to create a standalone unit." %
                (self, value))
        if self._workflow is not None:
            self._workflow.del_ref(self)
        self._workflow = value
        self._workflow.add_ref(self)

    def detach(self):
        self._workflow = weakref.proxy(self._workflow)

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
        Unit._pool_lock_.acquire()
        try:
            if Unit._pool_ is None:
                Unit._pool_ = thread_pool.ThreadPool(
                    minthreads=root.common.ThreadPool.minthreads,
                    maxthreads=root.common.ThreadPool.maxthreads)
        finally:
            Unit._pool_lock_.release()
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
    def run_was_called(self):
        return self._run_calls > 0

    @run_was_called.setter
    def run_was_called(self, value):
        if value:
            self._run_calls += 1
        else:
            raise ValueError("You can not reset run_was_called flag.")

    @property
    def total_run_time(self):
        return Unit.timers[self.id]

    @property
    def average_run_time(self):
        return self.total_run_time / self._run_calls \
            if self._run_calls > 0 else 0

    def stop(self):
        """By default, do nothing and consider run() to always finish.
        """
        pass

    def initialize_dependent(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for unit in self.dependent_list():
            unit.initialize()

    def run_dependent(self):
        """Invokes run() on dependent units on different threads.
        """
        for dst in self.links_to.keys():
            if dst.gate_block:
                continue
            if len(self.links_to) == 1:
                dst._check_gate_and_run(self)
            else:
                self.thread_pool.callInThread(dst._check_gate_and_run, self)

    def dependent_list(self, with_open_gate=False):
        units = [self]
        walk = []
        visited = {self}
        for child in sorted(self.links_to.keys()):
            walk.append((child, self))
        # flatten the dependency tree by doing breadth first search
        while len(walk) > 0:
            node, parent = walk.pop(0)
            if node in visited or (with_open_gate and
                                   not node.open_gate(parent)):
                continue
            units.append(node)
            visited.add(node)
            for child in sorted(node.links_to.keys()):
                walk.append((child, node))
        return units

    def open_gate(self, *args):
        """Called before run() or initialize().

        Returns:
            True: gate is open, can call run() or initialize().
            False: gate is closed, run() and initialize() shouldn't be called.
        """
        with self._gate_lock_:
            if not len(self.links_from):
                return True
            for src in args:
                if src in self.links_from:
                    self.links_from[src] = True
                if not all(self.links_from.values()):
                    return False
                for src in self.links_from:  # reset activation flags
                    self.links_from[src] = False
        return True

    def link_from(self, *args):
        """Adds notification link.
        """
        with self._gate_lock_:
            for src in args:
                self.links_from[src] = False
                with src._gate_lock_:
                    src.links_to[self] = False

    def unlink_from(self, *args):
        """Unlinks self from src.
        """
        with self._gate_lock_:
            for src in args:
                with src._gate_lock_:
                    if self in src.links_to:
                        del src.links_to[self]
                if src in self.links_from:
                    del self.links_from[src]

    def unlink_all(self):
        """Unlinks self from other units.
        """
        self.unlink_before()
        self.unlink_after()

    def unlink_before(self):
        """
        Detaches all previous units from this one.
        """
        with self._gate_lock_:
            for src in self.links_from:
                with src._gate_lock_:
                    del src.links_to[self]
            self.links_from.clear()

    def unlink_after(self):
        """
        Detaches all subsequent units from this one.
        """
        with self._gate_lock_:
            for dst in self.links_to:
                with dst._gate_lock_:
                    del dst.links_from[self]
            self.links_to.clear()

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
            for dst in links_to:
                with dst._gate_lock_:
                    dst.link_from(last)

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

    def nothing(self, *args, **kwargs):
        """Function that do nothing.

        It may be used to overload some methods to do nothing.
        """
        pass

    def describe(self):
        real_name = self.name if self._name is not None else "<not set>"
        res = "\n\033[1;36mUnit:\033[0m \"%s\"\n" % real_name
        res += "\033[1;36mClass:\033[0m %s.%s\n" % (self.__class__.__module__,
                                                    self.__class__.__name__)
        res += "\033[1;36mIncoming links:\033[0m\n"
        for link in self.links_from:
            res += "\t%s" % repr(link)
        res += "\n\033[1;36mOutgoing links:\033[0m\n"
        for link in self.links_to:
            res += "\t%s" % repr(link)
        print(res)

    @staticmethod
    def is_immutable(value):
        return (isinstance(value, tuple) or isinstance(value, int) or
                isinstance(value, float) or isinstance(value, complex) or
                isinstance(value, bool) or isinstance(value, str))

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
        if not self.open_gate(src):  # gate has a priority over skip
            return
        # Optionally skip the execution
        if not self.gate_skip:
            # If previous run has not yet finished, discard notification.
            if not self._run_lock_.acquire(False):
                return
            try:
                if not self._is_initialized:
                    self.error("%s is not initialized", self.name)
                    raise Unit.NotInitializedError()
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

    def _track_call(self, fn, name):
        def wrapped_track_call(*args, **kwargs):
            res = fn(*args, **kwargs)
            setattr(self, name, True)
            return res

        fnname = getattr(fn, '__name__',
                         getattr(fn, 'func', wrapped_track_call).__name__)
        wrapped_track_call.__name__ = fnname + '_track_call'
        return wrapped_track_call

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
    pass
