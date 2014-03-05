"""
Created on May 23, 2013

Random generators.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import threading
import formats


_lock = threading.Lock()


class Rand(object):
    """Random generator.

    Attributes:
        state: random state.
    """
    def __init__(self):
        if numpy.random.get_state is not None:
            self.state = numpy.random.get_state()

    def seed(self, seed, dtype=None, count=None):
        global _lock
        try:
            _lock.acquire()
            if numpy.random.get_state is not None:
                state = numpy.random.get_state()
            if type(seed) == str:
                fin = open(seed, "rb")
                seed = numpy.zeros(count, dtype=dtype)
                n = fin.readinto(seed)
                fin.close()
                seed = seed[:n // seed[0].nbytes]
            numpy.random.seed(seed)
            if numpy.random.get_state is not None:
                self.state = numpy.random.get_state()
                numpy.random.set_state(state)
        finally:
            _lock.release()

    def normal(self, loc=0.0, scale=1.0, size=None):
        """numpy.normal() with saving the random state.
        """
        global _lock
        _lock.acquire()
        self.save_state()
        retval = numpy.random.normal(loc=loc, scale=scale, size=size)
        self.restore_state()
        _lock.release()
        return retval

    def fill(self, arr, vle_min=-1.0, vle_max=1.0):
        """Fills numpy array with random numbers.

        Parameters:
            arr: numpy array.
            vle_min: minimum value in random distribution.
            vle_max: maximum value in random distribution.
        """
        global _lock
        _lock.acquire()
        self.save_state()
        arr = formats.ravel(arr)
        if arr.dtype in (numpy.complex64, numpy.complex128):
            # Fill the circle in case of complex numbers.
            r = numpy.random.rand(arr.size) * (vle_max - vle_min)
            a = numpy.random.rand(arr.size) * numpy.pi * 2.0
            arr.real[:] = r * numpy.cos(a)
            arr.imag[:] = r * numpy.sin(a)
        else:
            arr[:] = (numpy.random.rand(arr.size) * (vle_max - vle_min) +
                      vle_min)[:]
        self.restore_state()
        _lock.release()

    def fill_normal(self, arr, vle_min=-1.0, vle_max=1.0):
        """Fills numpy array with random numbers with normal distribution.

        Parameters:
            arr: numpy array.
            vle_min: minimum value in random distribution.
            vle_max: maximum value in random distribution.
        """
        global _lock
        _lock.acquire()
        self.save_state()
        arr = formats.ravel(arr)
        center = (vle_min + vle_max) * 0.5
        radius = (vle_max - vle_min) * 0.5
        if arr.dtype in (numpy.complex64, numpy.complex128):
            # Fill the circle in case of complex numbers.
            r = numpy.clip(numpy.random.normal(loc=center, scale=radius,
                size=arr.size), vle_min, vle_max)
            a = numpy.random.rand(arr.size) * numpy.pi * 2.0
            arr.real[:] = r * numpy.cos(a)
            arr.imag[:] = r * numpy.sin(a)
        else:
            arr[:] = numpy.clip(numpy.random.normal(loc=center, scale=radius,
                size=arr.size), vle_min, vle_max)[:]
        self.restore_state()
        _lock.release()

    def shuffle(self, arr):
        """numpy.shuffle() with saving the random state.
        """
        global _lock
        _lock.acquire()
        self.save_state()
        if numpy.random.shuffle is not None:
            numpy.random.shuffle(arr)
        else:
            import logging
            logging.warn("numpy.random.shuffle is None")
            n = len(arr) - 1
            for i in range(n):
                j = n + 1
                while j >= n + 1:  # pypy workaround
                    j = numpy.random.randint(i, n + 1)
                t = arr[i]
                arr[i] = arr[j]
                arr[j] = t
        self.restore_state()
        _lock.release()

    def permutation(self, x):
        """numpy.permutation() with saving the random state.
        """
        global _lock
        _lock.acquire()
        self.save_state()
        retval = numpy.random.permutation(x)
        self.restore_state()
        _lock.release()
        return retval

    def randint(self, low, high=None, size=None):
        """Returns random integer(s) from [low, high).
        """
        global _lock
        _lock.acquire()
        self.save_state()
        retval = numpy.random.randint(low, high, size)
        self.restore_state()
        _lock.release()
        return retval

    def rand(self, *args):
        global _lock
        _lock.acquire()
        self.save_state()
        retval = numpy.random.rand(*args)
        self.restore_state()
        _lock.release()
        return retval

    def save_state(self):
        if numpy.random.get_state is None:
            return
        self.saved_state = numpy.random.get_state()
        numpy.random.set_state(self.state)

    def restore_state(self):
        if numpy.random.get_state is None:
            return
        self.state = numpy.random.get_state()
        numpy.random.set_state(self.saved_state)


# Default global random instances.
default = Rand()
default2 = Rand()
