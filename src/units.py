"""
Created on Mar 12, 2013

Units in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import time
import numpy
import _thread


def realign(arr, boundary=4096):
    """Reallocates array to become PAGE-aligned as required for clEnqueueMapBuffer().
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
    """Will save attributes ending with _ as None when pickling and will call constructor upon unpickling.
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
        enabled: enabled unit or not.
        initialized: initialized unit or not.
        gate_lock_: lock.
        run_lock_: lock.
    """
    def __init__(self, unpickling=0):
        super(Unit, self).__init__(unpickling=unpickling)
        self.gate_lock_ = _thread.allocate_lock()
        self.run_lock_ = _thread.allocate_lock()
        self.initialized = 0
        if unpickling:
            return
        self.links_from = {}
        self.links_to = {}
        self.enabled = 1  # TODO(a.kazantsev): think about its purpose.

    def link_from(self, src):
        """Adds notification link.
        """
        self.links_from[src] = 0
        src.links_to[self] = 0

    def _initialize_dst(self, dst):
        """Initializes dst.
        """
        if not dst.gate(self):
            return
        self.run_lock_.acquire()
        if dst.initialize():
            self.run_lock_.release()
            return
        dst.initialized = 1
        self.run_lock_.release()
        dst.initialize_dependent()

    def _run_dst(self, dst):
        """Runs dst.
        """
        if not dst.gate(self):
            return
        self.run_lock_.acquire()
        # Initialize unit runtime if it is not initialized
        # TODO(a.kazantsev): or maybe raise an exception?
        if not dst.initialized:
            if dst.initialize():
                self.run_lock_.release()
                return
            dst.initialized = 1
        if dst.run():
            self.run_lock_.release()
            return
        self.run_lock_.release()
        dst.run_dependent()

    def initialize_dependent(self):
        """Invokes initialize() on dependent units.
        """
        for dst in self.links_to.keys():
            if dst.enabled and not dst.initialized:
                # _thread.start_new_thread(self._initialize_dst, (dst, ))
                self._initialize_dst(dst)  # there is no need to invoke it on different thread

    def run_dependent(self):
        """Invokes run() on dependent units.
        """
        for dst in self.links_to.keys():
            if dst.enabled:
                _thread.start_new_thread(self._run_dst, (dst,))

    def initialize(self):
        """Allocate buffers here.

        Returns:
            None: all ok, dependent units will be initialized.
            non-zero: error possibly occured, dependent units will not be initialized.
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
        cl_sources: OpenCL source files.
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
