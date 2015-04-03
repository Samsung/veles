"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 23, 2013

VELES reproducible random generators.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy
import os
import threading
from veles.compat import PYPY

from veles.config import root
from veles.distributable import Pickleable
from veles.numpy_ext import ravel


# Disable stock numpy.random for great justice
my_random = numpy.random


class WrappedRandom(object):
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(
            "veles.prng disables any direct usage of numpy.random. You can "
            "use veles.prng.get().%(item)s or self.prng.%(item)s instead." %
            locals())

numpy.random = WrappedRandom()


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
            my_random.seed(seed)
        except ValueError:
            my_random.seed(seed.view(numpy.uint32))
        if not PYPY:
            numpy.save(self.seed_file_name, seed)
        else:
            with open(self.seed_file_name, "wb") as fout:
                if isinstance(seed, numpy.ndarray):
                    fout.write(bytes(seed.data))
                else:
                    fout.write(bytes(numpy.asarray((seed,)).data))
        self.restore_state()

    def preserve_state(fn):
        def wrapped_preserve_state(self, *args, **kwargs):
            self.save_state()
            retval = fn(self, *args, **kwargs)
            self.restore_state()
            return retval

        wrapped_preserve_state.__name__ = "wrapped_" + fn.__name__
        return wrapped_preserve_state

    @threadsafe
    @preserve_state
    def normal(self, loc=0.0, scale=1.0, size=None):
        """numpy.normal() with saving the random state.
        """
        return my_random.normal(loc=loc, scale=scale, size=size)

    @threadsafe
    @preserve_state
    def uniform(self, low=0.0, high=1.0, size=None):
        return my_random.uniform(low=low, high=high, size=size)

    @threadsafe
    @preserve_state
    def random(self, size=None):
        return my_random.random(size=size)

    @threadsafe
    @preserve_state
    def choice(self, a, size=None, replace=True, p=None):
        return my_random.choice(a, size=size, replace=replace, p=p)

    @threadsafe
    @preserve_state
    def bytes(self, length):
        return my_random.bytes(length)

    @threadsafe
    @preserve_state
    def fill(self, arr, vle_min=-1.0, vle_max=1.0):
        """Fills numpy array with random numbers.

        Parameters:
            arr: numpy array.
            vle_min: minimum value in random distribution.
            vle_max: maximum value in random distribution.
        """
        arr = ravel(arr)
        arr[:] = (my_random.rand(arr.size) * (vle_max - vle_min) +
                  vle_min)[:]

    @threadsafe
    @preserve_state
    def fill_normal_real(self, arr, mean, stddev, clip_to_sigma=5.0):
        """
        #Fills real-valued numpy array with random normal distribution.

        #Parameters:
        #    arr: numpy array.
        #    mean:
        #    stddev:
        #    min_val, max_val (optional): clipping values of output data.
        """
        arr = ravel(arr)
        arr[:] = my_random.normal(loc=mean, scale=stddev, size=arr.size)[:]

        numpy.clip(arr, mean - clip_to_sigma * stddev,
                   mean + clip_to_sigma * stddev, out=arr)

    @threadsafe
    @preserve_state
    def shuffle(self, arr):
        """numpy.shuffle() with saving the random state.
        """
        if my_random.shuffle is not None:
            my_random.shuffle(arr)
        else:
            import logging
            logging.getLogger(self.__class__.__name__).warning(
                "numpy.random.shuffle is None")
            n = len(arr) - 1
            for i in range(n):
                j = n + 1
                while j >= n + 1:  # pypy workaround
                    j = my_random.randint(i, n + 1)
                t = arr[i]
                arr[i] = arr[j]
                arr[j] = t

    @threadsafe
    @preserve_state
    def permutation(self, x):
        """numpy.permutation() with saving the random state.
        """
        return my_random.permutation(x)

    @threadsafe
    @preserve_state
    def randint(self, low, high=None, size=None):
        """Returns random integer(s) from [low, high).
        """
        return my_random.randint(low, high, size)

    @threadsafe
    @preserve_state
    def random_sample(self, size=None):
        """Returns random integer(s) from [low, high).
        """
        return my_random.random_sample(size)

    @threadsafe
    @preserve_state
    def rand(self, *args):
        return my_random.rand(*args)

    def __call__(self, *args):
        return self.rand(*args)

    def save_state(self):
        if my_random.get_state is None:
            return
        self._saved_state = my_random.get_state()
        if self._state is not None:
            my_random.set_state(self._state)

    def restore_state(self):
        if my_random.get_state is None:
            return
        self._state = my_random.get_state()
        if self._saved_state is not None:
            my_random.set_state(self._saved_state)

    def _get_state(self):
        if my_random.get_state is None:
            return None
        return my_random.get_state()

    threadsafe = staticmethod(threadsafe)
    preserve_state = staticmethod(preserve_state)


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


# This is needed for scipy.stats.distributions
numpy.random.random_sample = get().random_sample

# This is harmless
numpy.random.get_state = get()._get_state
