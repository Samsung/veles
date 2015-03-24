"""
This file is a part of (almost) compeletely rewritten original txZmq project.

Created on Jul 3, 2014.

Buffer exchange via the shared memory.

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from mmap import mmap
from posix_ipc import SharedMemory, O_CREAT, ExistentialError


class SharedIO(object):
    """
    A version of BytesIO which is shared between multiple processes, suitable
    for IPC.
    """

    CACHE = {}

    def __init__(self, name, size):
        self.shmem = None
        self.shmem = SharedMemory(name, flags=O_CREAT, mode=0o666, size=size)
        self.file = mmap(self.shmem.fd, size)
        self.__init_file_methods()
        self.__shmem_refs = [1]

    def __del__(self):
        if self.shmem is None:
            return
        self.__shmem_refs[0] -= 1
        if self.__shmem_refs[0] == 0:
            self.shmem.close_fd()
            try:
                self.shmem.unlink()
            except ExistentialError:
                pass

    def __getstate__(self):
        return {"name": self.name,
                "size": self.size,
                "pos": self.tell()}

    def __setstate__(self, state):
        name = state["name"]
        size = state["size"]
        cached = SharedIO.CACHE.get("%s:%d" % (name, size))
        if not cached is None:
            assert cached.size == size
            self.shmem = cached.shmem
            self.file = cached.file
            self.__shmem_refs = cached.__shmem_refs
            self.__shmem_refs[0] += 1
            self.__init_file_methods()
        else:
            self.__init__(name, size)
            SharedIO.CACHE["%s:%d" % (name, size)] = self
        self.seek(state["pos"])

    @property
    def name(self):
        return self.shmem.name

    @property
    def size(self):
        return self.shmem.size

    @property
    def refs(self):
        return self.__shmem_refs[0]

    def __init_file_methods(self):
        for name in ("read", "readline", "write", "tell", "close", "seek"):
            setattr(self, name, getattr(self.file, name))
