"""
Created on Mar 12, 2013

Units in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import time
import logger
import threading
import thread_pool
import yaml


class Pickleable(logger.Logger):
    """Will save attributes ending with _ as None when pickling and will call
    constructor upon unpickling.
    """
    def __init__(self):
        """Calls init_unpickled() to initialize the attributes which are not
        pickled.
        """
        super(Pickleable, self).__init__()
        self.init_unpickled()

    """This function is called if the object has been just unpickled.
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


class Unit(Pickleable):
    """General unit in data stream model.

    Attributes:
        links_from: dictionary of units it depends on.
        links_to: dictionary of dependent units.
        initialized: initialized unit or not.
        gate_lock_: lock.
        run_lock_: lock.
        gate_block: if [0] is true, gate() and run() will not be executed and
                    notification will not be sent further
                    ([0] can be a function).
        gate_skip: if [0] is true, gate() and run() will not be executed, but
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

    def init_unpickled(self):
        super(Unit, self).init_unpickled()
        self.gate_lock_ = threading.Lock()
        self.run_lock_ = threading.Lock()
        self.initialized = 0

    def link_from(self, src):
        """Adds notification link.
        """
        self.links_from[src] = 0
        src.links_to[self] = 0

    def check_gate_and_initialize(self, src):
        """Check gate state and initialize if it is open.
        """
        if not self.gate(src):  # gate has a priority over skip
            return
        if ((callvle(self.gate_skip[0]) and
             (not callvle(self.gate_skip_not[0]))) or
            ((not callvle(self.gate_skip[0])) and
             (callvle(self.gate_skip_not[0])))):
            self.initialize_dependent()
            return
        self.run_lock_.acquire()
        if self.initialize():
            self.run_lock_.release()
            return
        self.initialized = 1
        self.run_lock_.release()
        self.initialize_dependent()

    def check_gate_and_run(self, src):
        """Check gate state and run if it is open.
        """
        if not self.gate(src):  # gate has a priority over skip
            return
        if ((callvle(self.gate_skip[0]) and
             (not callvle(self.gate_skip_not[0]))) or
            ((not callvle(self.gate_skip[0])) and
             callvle(self.gate_skip_not[0]))):
            self.run_dependent()
            return
        # If previous run has not yet executed, discard notification.
        if not self.run_lock_.acquire(False):
            return
        # Initialize unit runtime if it is not initialized.
        if not self.initialized:
            if self.initialize():
                self.run_lock_.release()
                return
            self.initialized = 1
        if self.run():
            self.run_lock_.release()
            return
        self.run_lock_.release()
        self.run_dependent()

    def initialize_dependent(self):
        """Invokes initialize() on dependent units on the same thread.
        """
        for dst in self.links_to.keys():
            if dst.initialized:
                continue
            if ((callvle(dst.gate_block[0]) and
                 (not callvle(dst.gate_block_not[0]))) or
                ((not callvle(dst.gate_block[0])) and
                 callvle(dst.gate_block_not[0]))):
                continue
            dst.check_gate_and_initialize(self)

    def run_dependent(self):
        """Invokes run() on dependent units on different threads.
        """
        for dst in self.links_to.keys():
            if ((callvle(dst.gate_block[0]) and
                 (not callvle(dst.gate_block_not[0]))) or
                ((not callvle(dst.gate_block[0])) and
                 callvle(dst.gate_block_not[0]))):
                continue
            thread_pool.pool.request(dst.check_gate_and_run, (self,))

    def initialize(self):
        """Allocate buffers here.

        initialize() invoked in the same order as run(), including gate() and
        effects of gate_block and gate_skip.

        If initialize() succedes, initialized flag will be automatically set.

        Returns:
            None: all ok, dependent units will be initialized.
            non-zero: error possibly occured, dependent units will not be
                      initialized.
        """
        pass

    def run(self):
        """Do the job here.

        Returns:
            None: all ok, dependent units will be run.
            non-zero: error possibly occured, dependent units will not be run.
        """
        pass

    def gate(self, src):
        """Called before run() or initialize().

        Returns:
            non-zero: gate is open, will invoke run() or initialize().
            zero: gate is closed, will not invoke further.
        """
        self.gate_lock_.acquire()
        if not len(self.links_from):
            self.gate_lock_.release()
            return 1
        if src in self.links_from:
            self.links_from[src] = 1
        if not all(self.links_from.values()):
            self.gate_lock_.release()
            return 0
        for src in self.links_from:  # reset activation flags
            self.links_from[src] = 0
        self.gate_lock_.release()
        return 1

    def unlink(self):
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

    def nothing(self):
        """Function that do nothing.
        """
        pass


class OpenCLUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        prg_: OpenCL program.
        cl_sources: OpenCL source files: file => defines.
    """
    def __init__(self, device=None):
        super(OpenCLUnit, self).__init__()
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

    def run(self):
        if not self.device:
            return self.cpu_run()
        return self.gpu_run()


class Repeater(Unit):
    """Repeater.
    """
    def gate(self, src):
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
