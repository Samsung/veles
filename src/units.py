"""
Created on Mar 12, 2013

Units in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import time
import numpy
import threading
import traceback
import sys


class ThreadPool(object):
    """Pool of threads.

    Attributes:
        sem_: semaphore.
        queue: queue of requests.
        total_threads: number of threads in the pool.
        free_threads: number of free threads in the pool.
    """
    def __init__(self, max_free_threads=10):
        self.sem_ = threading.Semaphore(0)
        self.lock_ = threading.Lock()
        self.exit_lock_ = threading.Lock()
        self.queue = []
        self.total_threads = 0
        self.free_threads = 0
        self.max_free_threads = max_free_threads
        self.exit_lock_.acquire()
        threading.Thread(target=self.pool_cleaner).start()
        self.sysexit = sys.exit
        sys.exit = self.exit

    def exit(self, retcode):
        self.shutdown()
        self.sysexit(retcode)


    def pool_cleaner(self):
        """Monitors request queue and executes requests,
            launching new threads if neccessary.
        """
        self.lock_.acquire()
        self.total_threads += 1
        self.lock_.release()
        while True:
            self.lock_.acquire()
            self.free_threads += 1
            if self.free_threads > self.max_free_threads:
                self.free_threads -= 1
                self.total_threads -= 1
                self.lock_.release()
                return
            self.lock_.release()
            try:
                self.sem_.acquire()
            except:  # return in case of broken semaphore
                self.lock_.acquire()
                self.free_threads -= 1
                self.total_threads -= 1
                if self.total_threads <= 0:
                    self.exit_lock_.release()
                self.lock_.release()
                return
            self.lock_.acquire()
            self.free_threads -= 1
            if self.free_threads <= 0:
                threading.Thread(target=self.pool_cleaner).start()
            try:
                (run, args) = self.queue.pop(0)
            except:
                self.total_threads -= 1
                self.lock_.release()
                return
            self.lock_.release()
            try:
                run(*args)
            except:
                # TODO(a.kazantsev): add good error handling here.
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)

    def request(self, run, args=()):
        """Adds request for execution to the queue.
        """
        self.lock_.acquire()
        self.queue.append((run, args))
        self.lock_.release()
        self.sem_.release()

    def shutdown(self):
        """Safely shutdowns thread pool.
        """
        sem_ = self.sem_
        self.sem_ = None
        self.lock_.acquire()
        for i in range(0, self.free_threads):
            sem_.release()
        self.lock_.release()
        self.exit_lock_.acquire()


pool = ThreadPool()


def realign(arr, boundary=4096):
    """Reallocates array to become PAGE-aligned as required for
        clEnqueueMapBuffer().
    """
    if arr == None:
        return None
    address = arr.__array_interface__["data"][0]
    if address % boundary == 0:
        return arr
    N = numpy.prod(arr.shape)
    d = arr.dtype
    tmp = numpy.empty(N * d.itemsize + boundary, dtype=numpy.uint8)
    address = tmp.__array_interface__["data"][0]
    offset = (boundary - address % boundary) % boundary
    newarr = tmp[offset:offset + N * d.itemsize]\
        .view(dtype=d)\
        .reshape(arr.shape, order="C")
    newarr[:] = arr[:]
    return newarr


def aligned_zeros(shape, boundary=4096, dtype=numpy.float32):
    """Allocates PAGE-aligned array required for clEnqueueMapBuffer().
    """
    N = numpy.prod(shape)
    d = numpy.dtype(dtype)
    tmp = numpy.zeros(N * d.itemsize + boundary, dtype=numpy.uint8)
    address = tmp.__array_interface__["data"][0]
    offset = (boundary - address % boundary) % boundary
    return tmp[offset:offset + N * d.itemsize]\
        .view(dtype=d)\
        .reshape(shape, order="C")


class SmartPickling(object):
    """Will save attributes ending with _ as None when pickling and will call
        constructor upon unpickling.
    """
    def __init__(self, unpickling=0):
        """Constructor.

        Parameters:
            unpickling: if 1, object is being created via unpickling.
        """
        pass

    def __getstate__(self):
        """What to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k[len(k) - 1] != "_":
                state[k] = v
            else:
                state[k] = None
        return state

    def __setstate__(self, state):
        """What to unpickle.
        """
        self.__dict__.update(state)
        self.__init__(unpickling=1)


class Connector(SmartPickling):
    """Connects unit attributes (data flow).

    Attributes:
        mtime: time of the last modification.
    """
    def __init__(self, unpickling=0):
        super(Connector, self).__init__(unpickling=unpickling)
        if unpickling:
            return
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


class Unit(SmartPickling):
    """General unit in data stream model.

    Attributes:
        links_from: dictionary of units it depends on.
        links_to: dictionary of dependent units.
        initialized: initialized unit or not.
        gate_lock_: lock.
        run_lock_: lock.
        gate_block: if true, gate() and run() will not be executed and
                    notification will not be sent further.
        gate_skip: if true, gate() and run() will not be executed, but
                   but notification will be sent further.
        gate_block_not: if true, inverses conditions for gate_block.
        gate_skip_not: if true, inverses conditions for gate_skip.
    """
    def __init__(self, unpickling=0):
        super(Unit, self).__init__(unpickling=unpickling)
        self.gate_lock_ = threading.Lock()
        self.run_lock_ = threading.Lock()
        self.initialized = 0
        if unpickling:
            return
        self.links_from = {}
        self.links_to = {}
        self.gate_block = [0]
        self.gate_skip = [0]
        self.gate_block_not = [0]
        self.gate_skip_not = [0]

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
        if (self.gate_skip[0] and not self.gate_skip_not[0]) or \
           (not self.gate_skip[0] and self.gate_skip_not[0]):
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
        if (self.gate_skip[0] and not self.gate_skip_not[0]) or \
           (not self.gate_skip[0] and self.gate_skip_not[0]):
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
            if (dst.gate_block[0] and not dst.gate_block_not[0]) or \
               (not dst.gate_block[0] and dst.gate_block_not[0]):
                continue
            dst.check_gate_and_initialize(self)

    def run_dependent(self):
        """Invokes run() on dependent units on different threads.
        """
        for dst in self.links_to.keys():
            if (dst.gate_block[0] and not dst.gate_block_not[0]) or \
               (not dst.gate_block[0] and dst.gate_block_not[0]):
                continue
            global pool
            pool.request(dst.check_gate_and_run, (self,))

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


class OpenCLUnit(Unit):
    """Unit that operates using OpenCL.

    Attributes:
        device: Device object.
        prg_: OpenCL program.
        cl_sources: OpenCL source files: file => defines.
    """
    def __init__(self, device=None, unpickling=0):
        super(OpenCLUnit, self).__init__(unpickling=unpickling)
        self.prg_ = None
        if unpickling:
            return
        self.device = device
        self.cl_sources = {}

    def cpu_run(self):
        """Run on CPU only.
        """
        pass

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
    def __init__(self, unpickling=0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)

    def run(self):
        self.sem_.release()

    def wait(self):
        self.sem_.acquire()
