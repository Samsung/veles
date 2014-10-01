#!/usr/bin/env python3

from copy import copy
import logging
import numpy
import opencl4py as cl
from pprint import pformat


class BufPoolError(RuntimeError):
    def __init__(self, msg):
        super(BufPoolError, self).__init__(msg)


class PoolStrategy(object):
    def __init__(self, mem_align=4096):
        self.mem_align = mem_align

    def create(self, vector, pool_size, regions):
        raise NotImplementedError("Failed to call PoolStrategy::create()")

    def release(self, vector, pool_size, regions):
        raise NotImplementedError("Failed to call PoolStrategy::release()")

    def pool_size(self, creation_dict):
        max_size = 0
        for vec in creation_dict:
            max_size = max(creation_dict[vec] + vec.nbytes, max_size)
        return max_size


class TrivialStrategy(PoolStrategy):
    def __init__(self, mem_align=4096):
        super(TrivialStrategy, self).__init__(mem_align)

    def create(self, vector, pool_size, regions):
        cur_pos = 0
        if len(regions) > 0:
            sizes = {}
            for vec in regions:
                sizes[regions[vec]["offset"]] = regions[vec]["size"]

            max_offset = sorted(sizes.keys())[-1]
            cur_pos = max_offset + sizes[max_offset] + self.mem_align
            cur_pos -= cur_pos % self.mem_align

        if pool_size - cur_pos >= vector.nbytes:
            regions[vector] = {"offset": cur_pos,
                               "size": vector.nbytes}
            return cur_pos
        else:
            raise BufPoolError("Failed to find free space in pool")

    def release(self, vector, pool_size, regions):
        pass


class FirstFitBuffer(PoolStrategy):
    def __init__(self, mem_align=4096):
        super(FirstFitBuffer, self).__init__(mem_align)

    def create(self, vector, pool_size, regions):
        cur_pos = 0
        if len(regions) > 0:
            sizes = {}
            for vec in regions:
                sizes[regions[vec]["offset"]] = regions[vec]["size"]

            for offset in sorted(sizes.keys()):
                if offset - cur_pos >= vector.nbytes:
                    regions[vector] = {"offset": cur_pos,
                                       "size": vector.nbytes}
                    return cur_pos
                cur_pos = offset + sizes[offset] + self.mem_align
                cur_pos -= cur_pos % self.mem_align

        if pool_size - cur_pos >= vector.nbytes:
            regions[vector] = {"offset": cur_pos, "size": vector.nbytes}
            return cur_pos
        else:
            raise BufPoolError("Failed to find free space in pool")

    def release(self, vector, pool_size, regions):
        del regions[vector]


class OclBufPool(object):
    def __init__(self, context):
        self._ctx = context
        self._vectors = {}
        self._host_buffer = None
        self._ocl_buffer = None
        self._pool_size = None

    def add(self, unit, vector):
        if self._vectors.get(unit) is not None:
            self._vectors[unit].add(vector)
        else:
            self._vectors[unit] = {vector}
        logging.debug("add %s (size = %d) from %s to bufpool", vector,
                      vector.nbytes, unit)

    def make_partitioning(self, start, strategy):
        # get list of future create and release actions
        action_list = self._build_action_list(start)
        logging.debug("action list:\n%s", pformat(action_list))

        # implement action list using specified strategy - get regions dict
        int64_max = numpy.iinfo(numpy.int64).max
        regions = {}
        creation_dict = {}
        for action in action_list:
            for vec in action["release"]:
                strategy.release(vec, int64_max, regions)
            for vec in action["create"]:
                creation_dict[vec] = strategy.create(vec, int64_max, regions)
                logging.debug("add to creation dict: offset = %d, size = %d",
                              creation_dict[vec], vec.nbytes)
        logging.debug("creation dict of sub-buffers:\n%s",
                      pformat(creation_dict))
        self._pool_size = strategy.pool_size(creation_dict)
        logging.debug("pool size: %d byte(s)", self._pool_size)

        # create host and ocl pooled buffer
        self._host_buffer = numpy.zeros(self._pool_size, dtype=numpy.uint8)
        self._host_buffer = cl.realign_array(
            self._host_buffer, strategy.mem_align, numpy)
        self._ocl_buffer = self._ctx.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, self._host_buffer)

        # create sub-buffers
        for vec in creation_dict:
            offset = creation_dict[vec]
            logging.debug("create sub-buffer for %s: offset = %d, size = %d",
                          vec, offset, vec.nbytes)
            saved = vec.mem
            start_pos = offset
            end_pos = start_pos + vec.nbytes
            vec.mem = self._host_buffer[start_pos:end_pos].view(
                dtype=saved.dtype)
            vec.mem.shape = saved.shape
            vec.mem[:] = saved[:]
            vec.devmem = self._ocl_buffer.create_sub_buffer(offset, vec.nbytes)

        self._vectors = creation_dict

    def _build_action_list(self, start):
        usage_list = []
        fwd_prop = []
        cum_set = set()

        unit = start
        # build list of objects in use
        while unit is not None:
            usage_list.append(self._vectors.get(unit, set()))
            cum_set.update(self._vectors.get(unit, set()))
            fwd_prop.append(copy(cum_set))

            if unit.links_to:
                # suppose that there is no branching in ocl pipeline
                if len(unit.links_to) > 1:
                    logging.warn(
                        "unit %s links to %d units (%s)",
                        unit.__class__, len(unit.links_to),
                        ', '.join([str(x.__class__) for x in unit.links_to]))
                unit = sorted(unit.links_to.keys())[0]
            else:
                unit = None
        logging.debug("buffers in use:\n%s", pformat(usage_list))
        logging.debug("forward propagation:\n%s", pformat(fwd_prop))

        # back propagation
        bwd_prop = copy(usage_list)
        for i in range(len(usage_list) - 1):
            bwd_prop[-i - 2].update(bwd_prop[-i - 1])
        logging.debug("backward propagation:\n%s", pformat(bwd_prop))

        # create action list
        if usage_list:
            action_list = [{"release": set(), "create": fwd_prop[0]}]
            for i in range(1, len(usage_list)):
                action_list.append({"release": bwd_prop[i - 1] - bwd_prop[i],
                                    "create": fwd_prop[i] - fwd_prop[i - 1]})
        return action_list
