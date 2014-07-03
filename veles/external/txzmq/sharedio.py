"""
Created on Jul 3, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from mmap import mmap
from posix_ipc import SharedMemory, O_CREAT, unlink_shared_memory


class SharedIO(object):
    """
    A version of BytesIO which is shared between multiple processes, suitable
    for IPC.
    """

    def __init__(self, name, size):
        self.shfd = SharedMemory(name, flags=O_CREAT, mode=0o666, size=size)
        self.file = mmap(self.fd, size)
        for name in ("read", "write", "tell", "close"):
            def method(self, *args, **kwargs):
                getattr(self.file, name)(*args, **kwargs)
            method.__name__ = name
            setattr(self, name, method)

    def __del__(self):
        unlink_shared_memory(self.shfd)
