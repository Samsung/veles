# -*- coding: utf-8 -*-
"""
Created on Mar 26, 2015

Unit which copies other units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from copy import deepcopy
import numpy
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit, \
    INumpyUnit
from veles.distributable import IDistributable, TriviallyDistributable
from veles.memory import Vector
from veles.mutable import Bool


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit, IDistributable)
class Avatar(AcceleratedUnit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "LOADER"
        super(Avatar, self).__init__(workflow, **kwargs)
        self._reals = {}
        self._vectors = {}
        self._remembers_gates = False

    @property
    def reals(self):
        return self._reals

    @property
    def vectors(self):
        return self._vectors

    def clone(self):
        for unit, attrs in self.reals.items():
            for attr in attrs:
                value = getattr(unit, attr)
                if self.is_immutable(value):
                    setattr(self, attr, value)
                    continue
                if not isinstance(value, Vector):
                    cloned = getattr(self, attr, None)
                    if cloned is None:
                        setattr(self, attr, deepcopy(value))
                        continue
                    if isinstance(value, list):
                        del cloned[:]
                        cloned.extend(value)
                    elif isinstance(value, (dict, set)):
                        cloned.clear()
                        cloned.update(value)
                    elif isinstance(value, Bool):
                        cloned <<= value
                    elif isinstance(value, numpy.ndarray):
                        cloned[:] = value
                    else:
                        setattr(self, attr, deepcopy(value))
                    continue
                vec = getattr(self, attr, None)
                if vec is None:
                    vec = Vector()
                    self.vectors[value] = vec
                    setattr(self, attr, vec)
                else:
                    assert isinstance(vec, Vector)
                if not vec and value:
                    vec.reset(value.mem.copy())

    def __getstate__(self):
        state = super(Avatar, self).__getstate__()
        for _unit, attrs in self.reals.items():
            for attr in attrs:
                if attr in state and self.is_immutable(getattr(self, attr)):
                    del state[attr]
        return state

    def initialize(self, device, **kwargs):
        super(Avatar, self).initialize(device, **kwargs)
        self.clone()
        self.init_vectors(*self.vectors.values())
        self.init_vectors(*self.vectors.keys())

    def run(self):
        self.clone()
        super(Avatar, self).run()

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            # Partial update
            return
        self.run()

    def generate_data_for_master(self):
        return True

    def ocl_init(self):
        pass

    def cuda_init(self):
        pass

    def ocl_run(self):
        for real, vec in self.vectors.items():
            real.unmap()
            vec.unmap()
            self.device.queue_.copy_buffer(
                real.devmem, vec.devmem, 0, 0, real.nbytes)

    def cuda_run(self):
        for real, vec in self.vectors.items():
            real.unmap()
            vec.unmap()
            vec.devmem.from_device_async(real.devmem)

    def numpy_run(self):
        for real, vec in self.vectors.items():
            real.map_read()
            vec.map_invalidate()
            numpy.copyto(vec.mem, real.mem)
