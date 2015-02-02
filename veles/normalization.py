"""
Created on Jan 28, 2015

Data normalization routines, classes and interfaces.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import numpy
import pickle
from PIL import Image
from six import add_metaclass
from zope.interface import implementer, Interface

from veles.compat import from_none
from veles.verified import Verified


class UninitializedStateError(Exception):
    pass


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


class NormalizerRegistry(type):
    """Metaclass to record Unit descendants. Used for introspection and
    analytical purposes.
    Classes derived from Unit may contain 'hide' attribute which specifies
    whether it should not appear in the list of registered units. Usually
    hide = True is applied to base units which must not be used directly, only
    subclassed. There is also a 'hide_all' attribute, do disable the
    registration of the whole inheritance tree, so that all the children are
    automatically hidden.
    """
    normalizers = {}

    def __init__(cls, name, bases, clsdict):
        yours = set(cls.mro())
        mine = set(object.mro())
        left = yours - mine
        if len(left) > 1 and "NAME" in clsdict:
            NormalizerRegistry.normalizers[clsdict["NAME"]] = cls
        super(NormalizerRegistry, cls).__init__(name, bases, clsdict)


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
                if k not in ("_initialized", "_cache")}

    def __setattr__(self, key, value):
        if getattr(self, "_initialized", False) and key not in self.__dict__:
            raise AttributeError(
                "Adding new attributes after initialize() was called is"
                "disabled.")
        super(NormalizerBase, self).__setattr__(key, value)


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

    NAME = "mean_disp"

    def _initialize(self, data):
        self._sum = numpy.zeros_like(data[0])
        self._count = 0
        self._min = numpy.array(data[0])
        self._max = numpy.array(data[0])

    def analyze(self, data):
        self._count += data.shape[0]
        self._sum += numpy.sum(data, axis=0)
        numpy.minimum(self._min, numpy.min(data, axis=0), self._min)
        numpy.maximum(self._max, numpy.max(data, axis=0), self._max)

    def _calculate_coefficients(self):
        return self._sum / self._count, self._max - self._min

    def normalize(self, data):
        mean, disp = self._calculate_coefficients()
        data -= mean
        data /= disp


@implementer(INormalizer)
class LinearNormalizer(StatelessNormalizer):
    """
    Normalizes values within the specified range from [min, max] in the current
    array *sample-wise*. Thus it is different from PointwiseNormalizer, which
    aggregates min and max through analyze() beforehand.
    """

    NAME = "linear"

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
        self._interval = vmin, vmax

    def normalize(self, data):
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        imin, imax = self.interval
        data *= (imin - imax) / (dmin - dmax)
        data += (dmin * imax - dmax * imin) / (dmin - dmax)


@implementer(INormalizer)
class ExponentNormalizer(StatelessNormalizer):
    """
    Subtracts the maximum from each sample, calculates the exponent and
    divides by the sum values in each sample. Thus each resulting subarray of
    exponent values gets distributed in (0, 1] and has the sum equal to 1.
    """

    NAME = "exp"

    def normalize(self, data):
        data -= data.max(axis=0)
        numpy.exp(data, data)
        data /= data.sum(axis=0)


@implementer(INormalizer)
class PointwiseNormalizer(NormalizerBase):
    """
    During the analysis stage, this class find the absolute minimum and maximum
    arrays of shape equal to the shape of input data. It then normalizes
    arrays within [-1, 1] from the found [min, max].
    """

    NAME = "pointwise"

    def _initialize(self, data):
        self._min = data[0].copy()
        self._max = data[0].copy()
        self._cache = [numpy.zeros_like(data[0]) for _ in range(2)]

    def analyze(self, data):
        numpy.minimum(self._min, numpy.min(data, axis=0), self._min)
        numpy.maximum(self._max, numpy.max(data, axis=0), self._max)

    def _calculate_coefficients(self):
        disp = self._max - self._min
        nzeros = numpy.nonzero(disp)

        mul, add = self._cache
        mul[nzeros] = 2.0
        mul[nzeros] /= disp[nzeros]
        mm = self._min * mul
        add[nzeros] = -1.0 - mm[nzeros]

        return mul, add

    def normalize(self, data):
        mul, add = self._calculate_coefficients()
        data *= mul
        data += add


@implementer(INormalizer)
class MeanFileNormalizer(StatelessNormalizer):
    """
    subtracts the "mean" sample from each subarray. Optionally, it then
    multiplies the result by scale.
    """

    NAME = "mean"

    def __init__(self, state=None, **kwargs):
        super(MeanFileNormalizer, self).__init__(state, **kwargs)
        self.scale = kwargs.get("scale", 1)
        # normalization_scale can be a reverse dispersion matrix
        if state is not None:
            return
        mean_path = kwargs["mean_path"]
        try:
            with open(mean_path, "rb") as fin:
                self.mean = numpy.array(Image.open(fin))
        except:
            try:
                self.mean = numpy.load(mean_path)
            except:
                try:
                    with open(mean_path, "rb") as fin:
                        self.mean = pickle.load(fin)
                except:
                    if isinstance(mean_path, numpy.ndarray):
                        self.mean = mean_path
                    else:
                        raise from_none(ValueError(
                            "Unable to load %s" % mean_path))
        if not isinstance(self.mean, numpy.ndarray):
            raise ValueError("%s is in invalid format" % mean_path)

    def normalize(self, data):
        data -= self.mean
        data *= self.scale
