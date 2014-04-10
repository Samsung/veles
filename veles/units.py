"""
Created on Mar 12, 2013

Units in data stream neural network_common model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


from copy import copy
from six.moves import cPickle as pickle
import threading
import time

from veles.config import root
import veles.error as error
import veles.logger as logger
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.thread_pool as thread_pool


class Pickleable(logger.Logger):
    """Will save attributes ending with _ as None when pickling and will call
    constructor upon unpickling.
    """
    def __init__(self, **kwargs):
        """Calls init_unpickled() to initialize the attributes which are not
        pickled.
        """
        self._method_storage = {}
        super(Pickleable, self).__init__(**kwargs)
        self.init_unpickled()

    """This function is called if the object has just been unpickled.
    """
    def init_unpickled(self):
        self.stripped_pickle_ = False
        for key, value in self._method_storage.items():
            setattr(self, key, getattr(value, key))

    def add_method_to_storage(self, name):
        self._method_storage[name] = self.__class__

    def __getstate__(self):
        """Selects the attributes to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k[len(k) - 1] != "_" and not callable(v):
                state[k] = v
            else:
                state[k] = None
        return state

    def __setstate__(self, state):
        """Recovers the object after unpickling.
        """
        self.__dict__.update(state)
        super(Pickleable, self).__init__()
        self.init_unpickled()

    @property
    def stripped_pickle(self):
        return self.stripped_pickle_

    @stripped_pickle.setter
    def stripped_pickle(self, value):
        self.stripped_pickle_ = value


class Distributable(Pickleable):
    def _data_threadsafe(self, fn):
        def wrapped(*args, **kwargs):
            self._data_lock_.acquire()
            result = None
            try:
                result = fn(*args, **kwargs)
            finally:
                self._data_lock_.release()
            return result

        return wrapped

    def __init__(self, **kwargs):
        self._generate_data_for_slave_threadsafe = \
            kwargs.get("generate_data_for_slave_threadsafe", False)
        self._apply_data_from_slave_threadsafe = \
            kwargs.get("apply_data_from_slave_threadsafe", True)
        super(Distributable, self).__init__(**kwargs)
        self.add_method_to_storage("generate_data_for_slave")
        self.add_method_to_storage("apply_data_from_slave")

    def init_unpickled(self):
        super(Distributable, self).init_unpickled()
        self._data_lock_ = threading.Lock()
        self._data_event_ = threading.Event()
        self._data_event_.set()
        if self._generate_data_for_slave_threadsafe:
            self.generate_data_for_slave = \
                self._data_threadsafe(self.generate_data_for_slave)
        if self._apply_data_from_slave_threadsafe:
            self.apply_data_from_slave = \
                self._data_threadsafe(self.apply_data_from_slave)
            self.drop_slave = \
                self._data_threadsafe(self.drop_slave)

    @property
    def has_data_for_slave(self):
        return self._data_event_.is_set()

    @has_data_for_slave.setter
    def has_data_for_slave(self, value):
        if value:
            self._data_event_.set()
        else:
            self._data_event_.clear()

    def wait_for_data_for_slave(self):
        self._data_event_.wait()

    """Callbacks for working in distributed environment.
    """
    def generate_data_for_master(self):
        """Data for master should be generated here. This function is executed
        on a slave instance.

        Returns:
            data of any type or None if there is nothing to send.
        """
        return None

    def generate_data_for_slave(self, slave=None):
        """Data for slave should be generated here. This function is executed
        on a master instance.
        This method is guaranteed to be threadsafe if
        generate_data_for_slave_threadsafe is set to True in __init__.

        Parameters:
            slave: some information about the slave (may be None).

        Returns:
            data of any type or None if there is nothing to send.
        """
        return None

    def apply_data_from_master(self, data):
        """Data from master should be applied here. This function is executed
        on a slave instance.

        Parameters:
            data - exactly the same value that was returned by
                   generate_data_for_slave at the master's side.

        Returns:
            None.
        """
        pass

    def apply_data_from_slave(self, data, slave=None):
        """Data from slave should be applied here. This function is executed
        on a master instance.
        This method is guaranteed to be threadsafe if
        apply_data_from_slave_threadsafe is set to True in __init__ (default).

        Parameters:
            slave: some information about the slave (may be None).

        Returns:
            None.
        """
        pass

    def drop_slave(self, slave=None):
        """Unexpected slave disconnection leads to this function call.
        This method is guaranteed to be threadsafe if
        apply_data_from_slave_threadsafe is set to True in __init__ (default).
        """
        pass

    def save(self, file_name):
        """
        Stores object's current state in the specified file.
        """
        data = self.generate_data_for_slave()
        pickle.dump(data, file_name)

    def load(self, file_name):
        """
        Loads object's current state from the specified file.
        """
        data = pickle.load(file_name)
        self.apply_data_from_master(data)


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
        self._links_from = {}
        self._links_to = {}
        self._gate_block = Bool(False)
        self._gate_skip = Bool(False)
        self._ran = False
        self._workflow = None
        self.workflow = workflow
        self.add_method_to_storage("initialize")
        self.add_method_to_storage("run")

    def init_unpickled(self):
        super(Unit, self).init_unpickled()
        self._gate_lock_ = threading.Lock()
        self._run_lock_ = threading.Lock()
        self._is_initialized = False
        self.initialize = self._dereference_attributes(self.initialize)
        self.run = self._track_call(self.run, "run_was_called")
        self.run = self._dereference_attributes(self.run)
        self.run = self._measure_time(self.run, Unit.timers)
        self.generate_data_for_slave = \
            self._dereference_attributes(self.generate_data_for_slave)
        self.generate_data_for_master = \
            self._dereference_attributes(self.generate_data_for_master)
        self.apply_data_from_slave = \
            self._dereference_attributes(self.apply_data_from_slave)
        self.apply_data_from_master = \
            self._dereference_attributes(self.apply_data_from_master)
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
    def view_group(self):
        return self._view_group

    @view_group.setter
    def view_group(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("Unit view group must be a string")
        self._view_group = value

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

    def initialize(self):
        """Allocate buffers here.

        initialize() invoked in the same order as run(), including
        open_gate() and effects of gate_block and gate_skip.

        If initialize() succeeds, self._is_initialized flag will be
        automatically set.
        """
        self._is_initialized = True

    def run(self):
        """Do the job here.
        """
        pass

    def initialize_dependent(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for dst in self.links_to.keys():
            if dst._is_initialized:
                continue
            if not dst.open_gate(self):
                continue
            dst.initialize()
            dst.initialize_dependent()

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

    def open_gate(self, src):
        """Called before run() or initialize().

        Returns:
            True: gate is open, can call run() or initialize().
            False: gate is closed, run() and initialize() shouldn't be called.
        """
        with self._gate_lock_:
            if not len(self.links_from):
                return True
            if src in self.links_from:
                self.links_from[src] = True
            if not all(self.links_from.values()):
                return False
            for src in self.links_from:  # reset activation flags
                self.links_from[src] = False
        return True

    def link_from(self, src):
        """Adds notification link.
        """
        with self._gate_lock_:
            self.links_from[src] = False
            with src._gate_lock_:
                src.links_to[self] = False

    def unlink_from(self, src):
        """Unlinks self from src.
        """
        with self._gate_lock_:
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
                    del(src.links_to[self])
            self.links_from.clear()

    def unlink_after(self):
        """
        Detaches all subsequent units from this one.
        """
        with self._gate_lock_:
            for dst in self.links_to:
                with dst._gate_lock_:
                    del(dst.links_from[self])
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

    def link_attrs(self, other, *args):
        """
        Assigns attributes from other to self, respecting whether each is
        mutable or immutable. In the latter case, an attribute link is created.
        """
        for arg in args:
            if (isinstance(arg, tuple) and len(arg) == 2 and
                    isinstance(arg[0], str) and isinstance(arg[1], str)):
                self._link_attr(other, *arg)
            elif isinstance(arg, str):
                self._link_attr(other, arg, arg)
            else:
                raise TypeError(repr(arg) + " is not a valid attributes pair")

    def def_attr(self, name, value):
        real = "_" + name
        setattr(self, real, value)
        setattr(self, name, (self, real))

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
    def is_attribute_reference(obj):
        return isinstance(obj, tuple) and len(obj) == 2 and \
            isinstance(obj[0], object) and isinstance(obj[1], str)

    def _link_attr(self, other, mine, yours):
        attr = getattr(other, yours)
        if (isinstance(attr, tuple) and not Unit.is_attribute_reference(attr))\
           or isinstance(attr, int) or isinstance(attr, float) \
           or isinstance(attr, bool) or isinstance(attr, str):
            setattr(self, mine, (other, yours))
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
            try:
                fn(*args, **kwargs)
            except TypeError as e:
                try:
                    fn(self, *args, **kwargs)
                except TypeError:
                    raise e
            fp = time.time()
            if self in storage:
                storage[self] += fp - sp

        return wrapped

    def _dereference_attributes(self, fn):
        """
        If any attribute of this class is a tuple (object, "name"), it is
        interpreted as a reference to object.name, so it is temporarily set to
        the dereferenced value before fn() call and then restored after fn() is
        finished, updating the referenced value.
        """
        def wrapped(*args, **kwargs):
            refs = {}
            for key, value in self.__dict__.items():
                if Unit.is_attribute_reference(value):
                    new_value = getattr(*value)
                    if Unit.is_attribute_reference(new_value):
                        raise RuntimeException()
                    setattr(self, key, new_value)
                    refs[key] = value
            try:
                fn(*args, **kwargs)
            except TypeError as e:
                try:
                    fn(self, *args, **kwargs)
                except TypeError:
                    raise e
            for key, value in refs.items():
                setattr(value[0], value[1], getattr(self, key))
                setattr(self, key, value)
        return wrapped

    def _track_call(self, fn, name):
        def wrapped(*args, **kwargs):
            setattr(self, name, True)
            try:
                fn(*args, **kwargs)
            except TypeError as e:
                try:
                    fn(self, *args, **kwargs)
                except TypeError:
                    raise e

        return wrapped


class OpenCLUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        prg_: OpenCL program.
        cl_sources: OpenCL source files: file => defines.
        prg_src: last built OpenCL program source code text.
    """
    def __init__(self, workflow, **kwargs):
        device = kwargs.get("device")
        kwargs["device"] = device
        super(OpenCLUnit, self).__init__(workflow, **kwargs)
        self.device = device

    def init_unpickled(self):
        super(OpenCLUnit, self).init_unpickled()
        self.prg_ = None
        self.cl_sources_ = {}

    def cpu_run(self):
        """Run on CPU only.
        """
        return super(OpenCLUnit, self).run()

    def gpu_run(self):
        """Run on GPU.
        """
        return self.cpu_run()

    def initialize(self, device=None):
        super(OpenCLUnit, self).initialize()
        if device is not None:
            self.device = device
        elif hasattr(self.workflow, "device"):
            self.device = self.workflow.device

    def run(self):
        t1 = time.time()
        if self.device:
            self.gpu_run()
        else:
            self.cpu_run()
        self.debug("%s in %.2f sec" %
                   (self.__class__.__name__, time.time() - t1))

    def build_program(self, defines=None, dump_filename=None, dtype=None):
        """Builds OpenCL program.

        prg_ will be initialized to the built program.
        """
        if defines and not isinstance(defines, dict):
            raise RuntimeError("defines must be a dictionary")
        lines = []
        my_defines = copy(defines) if defines else {}
        for fnme, defs in self.cl_sources_.items():
            lines.append("#include \"%s\"" % (fnme))
            my_defines.update(defs)
        if dtype is None:
            dtype = root.common.precision_type
        elif type(dtype) != str:
            dtype = opencl_types.numpy_dtype_to_opencl(dtype)
        my_defines.update(opencl_types.cl_defines[dtype])

        for k, v in my_defines.items():
            lines.insert(0, "#define %s %s" % (k, v))

        source = "\n".join(lines)

        try:
            self.prg_ = self.device.queue_.context.create_program(
                source, root.common.ocl_dirs)
        finally:
            if dump_filename is not None:
                flog = open(dump_filename, "w")
                flog.write(source)
                flog.close()

    def get_kernel(self, name):
        return self.prg_.get_kernel(name)

    def execute_kernel(self, krn, global_size, local_size):
        return self.device.queue_.execute_kernel(krn, global_size, local_size)


class Repeater(Unit):
    """Repeater.
    TODO(v.markovtsev): add more detailed description
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "PLUMBING")
        super(Repeater, self).__init__(workflow, **kwargs)

    def open_gate(self, src):
        """Gate is always open.
        """
        return True
