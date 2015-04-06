"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jan 28, 2015

Data normalization routines, classes and interfaces.

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
import pickle
from PIL import Image
from six import add_metaclass
from zope.interface import implementer, Interface

from veles.compat import from_none
from veles.numpy_ext import reshape, transpose
from veles.verified import Verified
from veles.unit_registry import MappedObjectsRegistry


class UninitializedStateError(Exception):
    pass


float32 = numpy.zeros(0, numpy.float32).dtype


class INormalizer(Interface):
    """
    Each normalization class must conform to this interface.
    """
    def __init__(state=None, **kwargs):  # pylint: disable=W0231
        """
        Initializes a new instance of INormalizer.
        :param state: The internal state (dict of instance attributes). If
        None, at least one call to analyze() must be made prior to normalize().
        :param kwargs: Additional options for the specific normalization type.
        """

    def analyze(data):
        """
        Analyze the given array, updating the internal state.
        :param data: numpy.ndarray
        """

    def normalize(data):
        """
        Performs the array values normalization based on the internal state.
        :param data: numpy.ndarray
        """

    def _initialize(data):
        """
        Initializes the internal state for the first time analyze() is called.
        This method is protected and should not be called by users.
        :param data: numpy.ndarray The array that was passed into analyze().
        """

    def _calculate_coefficients():
        """
        This method is protected and should not be called by users.
        Throws :class:`UninitializedStateError`
        in case :meth:`analyze()` has never been called before and no
        coefficients was supplied in class' constructor.
        :return: The recalculated normalization coefficients.
        """


class NormalizerRegistry(MappedObjectsRegistry):
    """Metaclass to record Unit descendants. Used for introspection and
    analytical purposes.
    Classes derived from Unit may contain 'hide' attribute which specifies
    whether it should not appear in the list of registered units. Usually
    hide = True is applied to base units which must not be used directly, only
    subclassed. There is also a 'hide_all' attribute, do disable the
    registration of the whole inheritance tree, so that all the children are
    automatically hidden.
    """
    mapping = "normalizers"
    base = object


@add_metaclass(NormalizerRegistry)
class NormalizerBase(Verified):
    """
    All normalization classes must inherit from this class.
    """

    def initialized(self, fn):
        def wrapped(data):
            if not self._initialized:
                self._initialize(data)
                self._initialized = True
            return fn(data)
        wrapped.__name__ = "initialized_" + fn.__name__
        return wrapped

    def assert_initialized(self, fn):
        def wrapped(data):
            assert self._initialized
            # assert data.dtype in (numpy.float32, numpy.float64) does not work
            # under PyPy, see https://bitbucket.org/pypy/pypy/issue/1998/numpyzeros-0-numpyfloat32-dtype-in  # nopep8
            assert data.dtype == numpy.float32 or data.dtype == numpy.float64
            assert isinstance(data.shape, tuple) and len(data.shape) > 1
            return fn(data)

        wrapped.__name__ = "assert_initialized_" + fn.__name__
        return wrapped

    def __init__(self, state=None, **kwargs):
        super(NormalizerBase, self).__init__(**kwargs)
        self.verify_interface(INormalizer)
        self._initialized = False
        self.analyze = self.initialized(self.analyze)
        self.normalize = self.assert_initialized(self.normalize)
        if state is not None:
            if not isinstance(state, dict):
                raise TypeError("state must be a dictionary")
            self.__dict__.update(state)
            self._initialized = True

    def analyze_and_normalize(self, data):
        self.analyze(data)
        self.normalize(data)

    @property
    def state(self):
        """
        Returns all instance attributes except _initialized and _cache.
        """
        assert self._initialized
        return {k: v for k, v in self.__dict__.items()
                if k not in ("_initialized", "_cache", "_logger_")
                and not hasattr(v, "__call__")}

    def __setattr__(self, key, value):
        if getattr(self, "_initialized", False) and key not in self.__dict__:
            raise AttributeError(
                "Adding new attributes after initialize() was called is"
                "disabled.")
        super(NormalizerBase, self).__setattr__(key, value)

    def __getstate__(self):
        state = self.state
        state.update(super(NormalizerBase, self).__getstate__())
        if "_cache" in state:
            del state["_cache"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        super(NormalizerBase, self).__setstate__(state)
        self.analyze = self.initialized(self.analyze)
        self.normalize = self.assert_initialized(self.normalize)

    def reset(self):
        if hasattr(self, "_cache"):
            delattr(self, "_cache")
        self._initialized = False

    @staticmethod
    def prepare(data):
        return transpose(reshape(
            data, (data.shape[0], data.size // data.shape[0])))


class StatelessNormalizer(NormalizerBase):
    """
    Special case of a normalizer without any internal state.
    """

    def __init__(self, state=None, **kwargs):
        super(StatelessNormalizer, self).__init__(state, **kwargs)
        # Strictly speaking, we could allow normalize() without analyze()
        # for suck stateless classes, but it would break the flow in case of
        # stateful normalizers and lead to errors after switching to them.
        # self._initialized = True

    def analyze(self, data):
        pass

    def _initialize(self, data):
        pass

    def _calculate_coefficients(self):
        return None


@implementer(INormalizer)
class MeanDispersionNormalizer(NormalizerBase):
    """
    Subtracts the mean value and divides by the difference between maximum and
    minimum. Please note that it does *not* divide by the dispersion as it is
    defined in statistics.
    """

    MAPPING = "mean_disp"

    def _initialize(self, data):
        # We force float64 to fix possible float32 saturation
        self._sum = numpy.zeros_like(data[0], dtype=numpy.float64)
        self._count = 0
        self._min = numpy.array(data[0])
        self._max = numpy.array(data[0])

    def analyze(self, data):
        self._count += data.shape[0]
        self._sum += numpy.sum(data, axis=0, dtype=numpy.float64)
        numpy.minimum(self._min, numpy.min(data, axis=0), self._min)
        numpy.maximum(self._max, numpy.max(data, axis=0), self._max)

    def _calculate_coefficients(self):
        return self._sum / self._count, self._max - self._min

    def normalize(self, data):
        mean, disp = self._calculate_coefficients()
        data -= mean
        disp[disp == 0] = 1
        data /= disp


@implementer(INormalizer)
class LinearNormalizer(StatelessNormalizer):
    """
    Normalizes values within the specified range from [min, max] in the current
    array *sample-wise*. Thus it is different from PointwiseNormalizer, which
    aggregates min and max through analyze() beforehand.
    """

    MAPPING = "linear"

    def __init__(self, state=None, **kwargs):
        super(LinearNormalizer, self).__init__(state, **kwargs)
        if state is None:
            self.interval = kwargs.get("interval", (-1, 1))

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        try:
            vmin, vmax = value
        except (TypeError, ValueError):
            raise from_none(ValueError("interval must consist of two values"))
        for v in vmin, vmax:
            if not isinstance(v, (int, float)):
                raise TypeError(
                    "Each value in the interval must be either an int or a "
                    "float (got %s of %s)" % (v, v.__class__))
        self._interval = float(vmin), float(vmax)

    def normalize(self, data):
        orig_shape_data = data
        data = NormalizerBase.prepare(data)
        dmin = numpy.min(data, axis=0)
        dmax = numpy.max(data, axis=0)
        imin, imax = self.interval
        diff = dmin - dmax
        if numpy.count_nonzero(diff) < numpy.prod(dmin.shape):
            self.warning("There are uniform samples and the normalization "
                         "type is linear, they are set to the middle %.2f "
                         "of the normalization interval [%.2f, %.2f]",
                         (imin + imax) / 2, imin, imax)
            zeros = diff == 0
            orig_shape_data[zeros] = (imin + imax) / 2
            dmax[zeros] = imax
            dmin[zeros] = imin
            diff[zeros] = imin - imax
        data *= (imin - imax) / diff
        data += (dmin * imax - dmax * imin) / diff


@implementer(INormalizer)
class ExponentNormalizer(StatelessNormalizer):
    """
    Subtracts the maximum from each sample, calculates the exponent and
    divides by the sum values in each sample. Thus each resulting subarray of
    exponent values gets distributed in (0, 1] and has the sum equal to 1.
    """

    MAPPING = "exp"

    def normalize(self, data):
        data = NormalizerBase.prepare(data)
        data -= data.max(axis=0)
        numpy.exp(data, data)
        data /= data.sum(axis=0)


@implementer(INormalizer)
class NoneNormalizer(StatelessNormalizer):
    """
    The most important normalizer which does simply nothing.
    """

    MAPPING = "none"

    def normalize(self, data):
        pass


@implementer(INormalizer)
class PointwiseNormalizer(NormalizerBase):
    """
    During the analysis stage, this class find the absolute minimum and maximum
    arrays of shape equal to the shape of input data. It then normalizes
    arrays within [-1, 1] from the found [min, max].
    """

    MAPPING = "pointwise"

    def _initialize(self, data):
        self._min = data[0].copy()
        self._max = data[0].copy()
        dtype = numpy.float32 if data[0].dtype != numpy.float64 \
            else numpy.float64
        self._cache = [numpy.zeros_like(data[0], dtype=dtype) for _ in (0, 1)]

    def analyze(self, data):
        numpy.minimum(self._min, numpy.min(data, axis=0), self._min)
        numpy.maximum(self._max, numpy.max(data, axis=0), self._max)

    def _calculate_coefficients(self):
        disp = self._max - self._min
        nonzeros = numpy.nonzero(disp)

        mul, add = self._cache
        mul[nonzeros] = 2.0
        mul[nonzeros] /= disp[nonzeros]
        mm = self._min * mul
        add[nonzeros] = -1.0 - mm[nonzeros]

        return mul, add

    def normalize(self, data):
        mul, add = self._calculate_coefficients()
        data *= mul
        data += add


class MeanNormalizerBase(object):
    def __init__(self, state=None, **kwargs):
        super(MeanNormalizerBase, self).__init__(state, **kwargs)
        self.scale = kwargs.get("scale", 1)

    @property
    def scale(self):
        """Can be a reversed dispersion matrix.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if not isinstance(value, (int, float, numpy.float32, numpy.float64)):
            raise TypeError("Scale must be a scalar floating point value")
        self._scale = float(value)

    def apply_scale(self, data):
        if self.scale != 1:
            data *= self.scale


@implementer(INormalizer)
class ExternalMeanNormalizer(MeanNormalizerBase, StatelessNormalizer):
    """
    Subtracts the "mean" sample from each subarray. Optionally, it then
    multiplies the result by scale.
    """

    MAPPING = "external_mean"

    def __init__(self, state=None, **kwargs):
        super(ExternalMeanNormalizer, self).__init__(state, **kwargs)
        if state is not None:
            return
        mean_source = kwargs["mean_source"]
        try:
            with open(mean_source, "rb") as fin:
                self.mean = numpy.array(Image.open(fin))
        except:
            try:
                self.mean = numpy.load(mean_source)
            except:
                try:
                    with open(mean_source, "rb") as fin:
                        self.mean = pickle.load(fin)
                except:
                    if isinstance(mean_source, numpy.ndarray):
                        self.mean = mean_source
                    else:
                        raise from_none(ValueError(
                            "Unable to load %s" % mean_source))
        if not isinstance(self.mean, numpy.ndarray):
            raise ValueError("%s is in invalid format" % mean_source)

    def normalize(self, data):
        data -= self.mean
        self.apply_scale(data)


@implementer(INormalizer)
class InternalMeanNormalizer(MeanNormalizerBase, NormalizerBase):
    """
    Subtracts the calculated globally "mean" sample from each subarray.
    Optionally, it then multiplies the result by scale.
    """

    MAPPING = "internal_mean"

    def analyze(self, data):
        self._count += data.shape[0]
        self._sum += numpy.sum(data, axis=0, dtype=numpy.float64)

    def _initialize(self, data):
        self._sum = numpy.zeros_like(data[0], dtype=numpy.float64)
        self._count = 0

    def _calculate_coefficients(self):
        return self._sum / self._count

    def normalize(self, data):
        data -= self._calculate_coefficients()
        self.apply_scale(data)
