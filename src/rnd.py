"""
Created on May 23, 2013

Random generators.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import _thread


_lock = _thread.allocate_lock()


class Rand(object):
    """Random generator.

    Attributes:
        state: random state.
    """
    def __init__(self):
        self.state = None

    def seed(self, seed):
        global _lock
        _lock.acquire()
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        self.state = numpy.random.get_state()
        numpy.random.set_state(state)
        _lock.release()

    def fill(self, arr, vle_min=-1.0, vle_max=1.0):
        """Fills numpy array with random numbers.

        Parameters:
            arr: numpy array.
            vle_min: minimum value in random distribution.
            vle_max: maximum value in random distribution.
        """
        global _lock
        _lock.acquire()
        state = numpy.random.get_state()
        numpy.random.set_state(self.state)
        arr = arr.reshape(arr.size)
        arr[:] = (numpy.random.rand(arr.size) * (vle_max - vle_min) + \
                  vle_min)[:]
        self.state = numpy.random.get_state()
        numpy.random.set_state(state)
        _lock.release()

    def shuffle(self, arr):
        """numpy.shuffle() with saving the random state.
        """
        global _lock
        _lock.acquire()
        state = numpy.random.get_state()
        numpy.random.set_state(self.state)
        numpy.random.shuffle(arr)
        self.state = numpy.random.get_state()
        numpy.random.set_state(state)
        _lock.release()

    def permutation(self, x):
        """numpy.permutation() with saving the random state.
        """
        global _lock
        _lock.acquire()
        state = numpy.random.get_state()
        numpy.random.set_state(self.state)
        retval = numpy.random.permutation(x)
        self.state = numpy.random.get_state()
        numpy.random.set_state(state)
        _lock.release()
        return retval


# Default global random instance.
default = Rand()
