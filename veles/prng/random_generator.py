"""
Created on May 23, 2013

VELES reproducible random generators.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import threading

from veles.config import root
from veles.distributable import Pickleable
import veles.memory as formats


class RandomGenerator(Pickleable):
    """Random generator with exact reproducibility property.

    Attributes:
        state: the random generator state.
    """

    _lock = threading.Lock()

    def threadsafe(fn):
        def wrapped(*args, **kwargs):
            with RandomGenerator._lock:
                res = fn(*args, **kwargs)
            return res
        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_threadsafe'
        return wrapped

    def __init__(self, key):
        super(RandomGenerator, self).__init__()
        self._key = key
        self._saved_state = None
        self.restore_state()

    @property
    def key(self):
        return self._key

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, vle):
        self._state = vle

    @property
    def seed_file_name(self):
        return os.path.join(root.common.cache_dir,
                            "random_seed_%s.npy" % str(self.key))

    @threadsafe
    def seed(self, seed, dtype=None, count=None):
        self._state = None
        self.save_state()
        if seed is None:
            seed = numpy.fromfile(self.seed_file_name)
        elif isinstance(seed, str):
            if not os.path.exists(seed):
                raise ValueError("No such file - %s" % seed)
            with open(seed, "rb") as fin:
                seed = numpy.zeros(count, dtype=dtype)
                n = fin.readinto(seed)
            seed = seed[:n // seed[0].nbytes]
        try:
            numpy.random.seed(seed)
        except ValueError:
            numpy.random.seed(seed.view(numpy.uint32))
        numpy.save(self.seed_file_name, seed)
        self.restore_state()

    @threadsafe
    def normal(self, loc=0.0, scale=1.0, size=None):
        """numpy.normal() with saving the random state.
        """
        self.save_state()
        retval = numpy.random.normal(loc=loc, scale=scale, size=size)
        self.restore_state()
        return retval

    @threadsafe
    def fill(self, arr, vle_min=-1.0, vle_max=1.0):
        """Fills numpy array with random numbers.

        Parameters:
            arr: numpy array.
            vle_min: minimum value in random distribution.
            vle_max: maximum value in random distribution.
        """
        self.save_state()
        arr = formats.ravel(arr)
        arr[:] = (numpy.random.rand(arr.size) * (vle_max - vle_min) +
                  vle_min)[:]
        self.restore_state()

    @threadsafe
    def fill_normal_real(self, arr, mean, stddev, clip_to_sigma=5.0):
        """
        #Fills real-valued numpy array with random normal distribution.

        #Parameters:
        #    arr: numpy array.
        #    mean:
        #    stddev:
        #    min_val, max_val (optional): clipping values of output data.
        """
        self.save_state()
        arr = formats.ravel(arr)
        arr[:] = numpy.random.normal(loc=mean, scale=stddev, size=arr.size)[:]

        numpy.clip(arr, mean - clip_to_sigma * stddev,
                   mean + clip_to_sigma * stddev, out=arr)
        self.restore_state()

    @threadsafe
    def shuffle(self, arr):
        """numpy.shuffle() with saving the random state.
        """
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

    @threadsafe
    def permutation(self, x):
        """numpy.permutation() with saving the random state.
        """
        self.save_state()
        retval = numpy.random.permutation(x)
        self.restore_state()
        return retval

    @threadsafe
    def randint(self, low, high=None, size=None):
        """Returns random integer(s) from [low, high).
        """
        self.save_state()
        retval = numpy.random.randint(low, high, size)
        self.restore_state()
        return retval

    @threadsafe
    def rand(self, *args):
        self.save_state()
        retval = numpy.random.rand(*args)
        self.restore_state()
        return retval

    def __call__(self, *args):
        return self.rand(*args)

    def save_state(self):
        if numpy.random.get_state is None:
            return
        self._saved_state = numpy.random.get_state()
        if self._state is not None:
            numpy.random.set_state(self._state)

    def restore_state(self):
        if numpy.random.get_state is None:
            return
        self._state = numpy.random.get_state()
        if self._saved_state is not None:
            numpy.random.set_state(self._saved_state)

    threadsafe = staticmethod(threadsafe)


def xorshift128plus(states, index):
    seed = states[index:index + 2]
    s1 = seed[0:1].copy()
    s0 = seed[1:2].copy()
    seed[0] = s0[0]
    s1 ^= s1 << 23  # a
    seed[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))[0]
    output = seed[1] + s0[0]  # b, c
    states[index:index + 2] = seed
    return output


__generators__ = {}


# Default global random instances.
def get(key=1):
    res = __generators__.get(key)
    if res is None:
        res = RandomGenerator(key)
        __generators__[key] = res
    return res
