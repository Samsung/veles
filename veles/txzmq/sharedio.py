# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 3, 2014.

This file is a part of (almost) compeletely rewritten original txZmq project.

Buffer exchange via the shared memory.

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
        if cached is not None:
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
