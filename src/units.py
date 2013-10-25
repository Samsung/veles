"""
Created on Mar 12, 2013

Units in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import time
import logger
import threading
import thread_pool
import config
import pyopencl


class Pickleable(logger.Logger):
    """Will save attributes ending with _ as None when pickling and will call
    constructor upon unpickling.
    """
    def __init__(self):
        """Calls init_unpickled() to initialize the attributes which are not
        pickled.
        """
        super(Pickleable, self).__init__()
        # self.init_unpickled()  # already called in Logger()

    """This function is called if the object has just been unpickled.
    """
    def init_unpickled(self):
        if hasattr(super(Pickleable, self), "init_unpickled"):
            super(Pickleable, self).init_unpickled()

    def __getstate__(self):
        """Selects the attributes to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            w = [None] if type(v) == list and len(v) == 1 and callable(v[0]) \
                else v
            if k[len(k) - 1] != "_":
                state[k] = w
            else:
                state[k] = None
        return state

    def __setstate__(self, state):
        """Recovers the object after unpickling.
        """
        self.__dict__.update(state)
        self.init_unpickled()


class Distributable():
    def generate_data_for_master(self):
        return None

    def generate_data_for_slave(self):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data):
        pass


class Connector(Pickleable):
    """Connects unit attributes (data flow).

    Attributes:
        mtime: time of the last modification.
    """
    def __init__(self):
        super(Connector, self).__init__()
        self.mtime = 0.0

    def update(self):
        """Marks data as updated (updates mtime).
        """
        mtime = time.time()
        if mtime <= self.mtime:
            dt = 0.000001
            mtime = self.mtime + dt
            while mtime <= self.mtime:
                mtime += dt
                dt += dt
        self.mtime = mtime


def callvle(var):
    return var() if callable(var) else var


global_lock = threading.Lock()
pool = None


class Unit(Pickleable, Distributable):
    """General unit in data stream model.

    Attributes:
        links_from: dictionary of units it depends on.
        links_to: dictionary of dependent units.
        is_initialized: is_initialized unit or not.
        gate_lock_: lock.
        run_lock_: lock.
        gate_block: if [0] is true, open_gate() and run() will not be executed and
                    notification will not be sent further
                    ([0] can be a function).
        gate_skip: if [0] is true, open_gate() and run() will not be executed, but
                   but notification will be sent further
                   ([0] can be a function).
        gate_block_not: if [0] is true, inverses conditions for gate_block
                   ([0] can be a function).
        gate_skip_not: if [0] is true, inverses conditions for gate_skip
                   ([0] can be a function).
        exports: list of attribute names to export
                 (None - unit is not exportable).
    """
    def __init__(self):
        super(Unit, self).__init__()
        self.links_from = {}
        self.links_to = {}
        self.gate_block = [0]
        self.gate_skip = [0]
        self.gate_block_not = [0]
        self.gate_skip_not = [0]
        self.exports = None
        self.applied_data_from_master_recursively = False
        self.applied_data_from_slave_recursively = False

    def init_unpickled(self):
        global global_lock
        global pool
        global_lock.acquire()
        try:
            if pool == None:
                pool = thread_pool.ThreadPool()
        finally:
            global_lock.release()
        super(Unit, self).init_unpickled()
        self.gate_lock_ = threading.Lock()
        self.run_lock_ = threading.Lock()
        self.is_initialized = False

    def link_from(self, src):
        """Adds notification link.
        """
        self.links_from[src] = False
        src.links_to[self] = False

    def check_gate_and_initialize(self, src):
        """Check gate state and initialize if it is open.
        """
        if not self.open_gate(src):  # gate has a priority over skip
            return
        # Optionally skip the execution
        if ((callvle(self.gate_skip[0]) and
             (not callvle(self.gate_skip_not[0]))) or
            ((not callvle(self.gate_skip[0])) and
             (callvle(self.gate_skip_not[0])))):
            self.initialize_recursively()
            return
        self.run_lock_.acquire()
        try:
            self.initialize()
            self.is_initialized = True
        except:
            return
        finally:
            self.run_lock_.release()
        self.initialize_recursively()

    def check_gate_and_run(self, src):
        """Check gate state and run if it is open.
        """
        if not self.open_gate(src):  # gate has a priority over skip
            return
        # Optionally skip the execution
        if ((callvle(self.gate_skip[0]) and
             (not callvle(self.gate_skip_not[0]))) or
            ((not callvle(self.gate_skip[0])) and
             callvle(self.gate_skip_not[0]))):
            self.run_recursively()
            return
        # If previous run has not yet finished, discard notification.
        if not self.run_lock_.acquire(blocking=False):
            return
        try:
            if not self.is_initialized:
                self.initialize()
                self.is_initialized = True
            self.run()
        except:
            return
        finally:
            self.run_lock_.release()
        self.run_recursively()

    def initialize_recursively(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for dst in self.links_to.keys():
            if dst.is_initialized:
                continue
            if ((callvle(dst.gate_block[0]) and
                 (not callvle(dst.gate_block_not[0]))) or
                ((not callvle(dst.gate_block[0])) and
                 callvle(dst.gate_block_not[0]))):
                continue
            dst.check_gate_and_initialize(self)

    def run_recursively(self):
        """Invokes run() on dependent units on different threads.
        """
        global pool
        for dst in self.links_to.keys():
            if ((callvle(dst.gate_block[0]) and
                 (not callvle(dst.gate_block_not[0]))) or
                ((not callvle(dst.gate_block[0])) and
                 callvle(dst.gate_block_not[0]))):
                continue
            pool.request(dst.check_gate_and_run, (self,))

    def initialize(self):
        """Allocate buffers here.

        initialize() invoked in the same order as run(), including
        open_gate() and effects of gate_block and gate_skip.

        If initialize() succeeds, self.is_initialized flag will be automatically
        set.
        """
        pass

    def run(self):
        """Do the job here.
        """
        pass

    def open_gate(self, src):
        """Called before run() or initialize().

        Returns:
            True: gate is open, can call run() or initialize().
            False: gate is closed, run() and initialize() shouldn't be called.
        """
        self.gate_lock_.acquire()
        if not len(self.links_from):
            self.gate_lock_.release()
            return True
        if src in self.links_from:
            self.links_from[src] = True
        if not all(self.links_from.values()):
            self.gate_lock_.release()
            return False
        for src in self.links_from:  # reset activation flags
            self.links_from[src] = False
        self.gate_lock_.release()
        return True

    def unlink_all(self):
        """Unlinks self from other units.
        """
        self.gate_lock_.acquire()
        for src in self.links_from:
            del(src.links_to[self])
        for dst in self.links_to:
            del(dst.links_from[self])
        self.links_from.clear()
        self.links_to.clear()
        self.gate_lock_.release()

    def unlink_from(self, src):
        """Unlinks self from src.
        """
        self.gate_lock_.acquire()
        if self in src.links_to:
            del src.links_to[self]
        if src in self.links_from:
            del self.links_from[src]
        self.gate_lock_.release()

    def generate_data_for_master_recursively(self):
        self.applied_data_from_master_recursively = False
        data = self.generate_data_for_master()
        for dst in self.links_to.keys():
            data = data + dst.generate_data_for_master_recursively()

    def generate_data_for_slave_recursively(self):
        self.applied_data_from_slave_recursively = False
        data = self.generate_data_for_slave()
        for dst in self.links_to.keys():
            data = data + dst.generate_data_for_slave_recursively()

    def apply_data_from_master_recursively(self, data):
        if not self.applied_data_from_master_recursively:
            self.apply_data_from_master(data[0])
            self.applied_data_from_master_recursively = True
            data = data[1:]
        for dst in self.links_to.keys():
            dst.apply_data_from_master_recursively(data)

    def apply_data_from_slave_recursively(self, data):
        if not self.applied_data_from_slave_recursively:
            self.apply_data_from_slave(data[0])
            self.applied_data_from_slave_recursively = True
            data = data[1:]
        for dst in self.links_to.keys():
            dst.apply_data_from_slave_recursively(data)


class OpenCLUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        prg_: OpenCL program.
        cl_sources: OpenCL source files: file => defines.
        prg_src: last built OpenCL program source code text.
    """
    def __init__(self, device=None):
        super(OpenCLUnit, self).__init__()
        self.device = device
        self.prg_src = None

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

    def run(self):
        t1 = time.time()
        if self.device:
            self.gpu_run()
        else:
            self.cpu_run()
        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                             time.time() - t1))

    def build_program(self, defines, log_fnme=None):
        """Builds OpenCL program.

        _prg will be is_initialized with built program.

        Parameters:
            defines: additional definitions.

        Returns:
            Built OpenCL program source code.
        """
        s = defines
        fin = open("%s/defines.cl" % (config.cl_dir), "r")
        s += fin.read()
        fin.close()
        for src, define in self.cl_sources_.items():
            s += "\n" + define + "\n"
            fin = open(src, "r")
            s += fin.read()
            fin.close()
        fin = open("%s/matrix_multiplication.cl" % (config.cl_dir), "r")
        s_mx_mul = fin.read()
        fin.close()
        s = s.replace("MX_MUL", s_mx_mul)
        if log_fnme != None:
            flog = open(log_fnme, "w")
            flog.write(s)
            flog.close()
        self.prg_src = s
        self.prg_ = pyopencl.Program(self.device.context_, s).build()


class Repeater(Unit):
    """Repeater.
    """
    def open_gate(self, src):
        """Gate is always open.
        """
        return 1


class EndPoint(Unit):
    """End point with semaphore.

    Attributes:
        sem_: semaphore.
    """
    def init_unpickled(self):
        super(EndPoint, self).init_unpickled()
        self.sem_ = threading.Semaphore(0)

    def run(self):
        self.sem_.release()

    def wait(self):
        self.sem_.acquire()
