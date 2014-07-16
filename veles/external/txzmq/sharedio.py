"""
Created on Jul 3, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from mmap import mmap
from posix_ipc import SharedMemory, O_CREAT, ExistentialError


class SharedIO(object):
    """
    A version of BytesIO which is shared between multiple processes, suitable
    for IPC.
    """

    def __init__(self, name, size):
        self.shmem = SharedMemory(name, flags=O_CREAT, mode=0o666, size=size)
        self.file = mmap(self.shmem.fd, size)
        for name in ("read", "write", "tell", "close", "seek"):
            setattr(self, name, getattr(self.file, name))

    def __del__(self):
        self.shmem.close_fd()
        try:
            self.shmem.unlink()
        except ExistentialError:
            pass
