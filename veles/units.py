"""
Created on Mar 12, 2013

Units in data stream neural network_common model.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import threading
import time
import uuid
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject

from veles.config import root, get
from veles.distributable import Distributable, TriviallyDistributable, \
    IDistributable
import veles.error as error
from veles.mutable import Bool, LinkableAttribute
import veles.thread_pool as thread_pool


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


class Unit(Distributable):
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

    def __init__(self, workflow, **kwargs):
        self.name = kwargs.get("name")
        self.view_group = kwargs.get("view_group")
        super(Unit, self).__init__(**kwargs)
        self.verify_interface(IUnit)
        self._links_from = {}
        self._links_to = {}
        self._gate_block = Bool(False)
        self._gate_skip = Bool(False)
        self._ran = False
        self._timings = kwargs.get("timings", get(root.common.timings, False))
        self._workflow = None
        self.workflow = workflow
        self._id = str(uuid.uuid4())
        self.add_method_to_storage("initialize")
        self.add_method_to_storage("run")

    def init_unpickled(self):
        def wrap_to_measure_time(name):
            func = getattr(self, name, None)
            if func is not None:
                setattr(self, name, self._measure_time(func, Unit.timers))

        # Important: these four decorator applications must stand before
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
        Unit.timers[self] = 0

    def __getstate__(self):
        state = super(Unit, self).__getstate__()
        if self.stripped_pickle:
            state["_links_from"] = {}
            state["_links_to"] = {}
            state["_workflow"] = None
        return state

    def __repr__(self):
        if self._name is not None:
            return "%s.%s \"%s\"" % (self.__class__.__module__,
                                     self.__class__.__name__,
                                     self.name)
        else:
            return object.__repr__(self)

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
        if self._workflow is not None:
            self._workflow.del_ref(self)
        self._workflow = value
        self._workflow.add_ref(self)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    @name.setter
    def name(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("Unit name must be a string")
        self._name = value

    @property
    def id(self):
        return self._id

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
        return self._ran

    @run_was_called.setter
    def run_was_called(self, value):
        self._ran = value

    def stop(self):
        """By default, do nothing and consider run() to always finish.
        """
        pass

    def initialize_dependent(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for unit in self.dependecy_list():
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

    def dependecy_list(self):
        units = [self]
        walk = []
        visited = {self}
        for child in self.links_to.keys():
            walk.append((child, self))
        # flatten the dependency tree by doing breadth first search
        while len(walk) > 0:
            node, parent = walk.pop(0)
            if node in visited or not node.open_gate(parent):
                continue
            units.append(node)
            visited.add(node)
            for child in node.links_to.keys():
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

    def verify_interface(self, iface):
        if not iface.providedBy(self):
            raise NotImplementedError(
                "Unit %s does not implement %s interface" % (repr(self),
                                                             iface.__name__))
        verifyObject(iface, self)

    def _link_attr(self, other, mine, yours, two_way):
        attr = getattr(other, yours)
        if (isinstance(attr, tuple) or isinstance(attr, int) or
                isinstance(attr, float) or isinstance(attr, complex) or
                isinstance(attr, bool) or isinstance(attr, str)):
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
                    self.initialize()
                    self.warning("%s was not initialized, performed the "
                                 "initialization", self.name)
                self.run()
            finally:
                self._run_lock_.release()
        self.run_dependent()

    def _measure_time(self, fn, storage):
        def wrapped(*args, **kwargs):
            sp = time.time()
            res = fn(*args, **kwargs)
            fp = time.time()
            delta = fp - sp
            if self in storage:
                storage[self] += delta
            if self.timings:
                self.debug("%s took %.2f sec", fn.__name__, delta)
            return res

        return wrapped

    def _track_call(self, fn, name):
        def wrapped(*args, **kwargs):
            res = fn(*args, **kwargs)
            setattr(self, name, True)
            return res

        return wrapped


@implementer(IUnit, IDistributable)
class TrivialUnit(Unit, TriviallyDistributable):
    def initialize(self, **kwargs):
        pass

    def run(self):
        pass
