

import six
from snakebite.client import Client
from zope.interface import implementer

from veles.distributable import TriviallyDistributable
from veles.loader.base import UserLoaderRegistry
from veles.mutable import Bool
from veles.units import IUnit, Unit


@six.add_metaclass(UserLoaderRegistry)
@implementer(IUnit)
class HDFSTextLoader(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        super(HDFSTextLoader, self).__init__(workflow, **kwargs)
        self.file_name = kwargs["file"]
        self.chunk_lines_number = kwargs.get("chunk", 1000)
        client_kwargs = dict(kwargs)
        del client_kwargs["file"]
        if "chunk" in kwargs:
            del client_kwargs["chunk"]
        self.hdfs_client = Client(**client_kwargs)
        self.output = [""] * self.chunk_lines_number
        self.finished = Bool()

    def initialize(self):
        self.debug("Opened %s", self.hdfs_client.stat([self.file_name]))
        self._generator = self.hdfs_client.text([self.file_name])

    def run(self):
        assert not self.finished
        try:
            for i in range(self.chunk_lines_number):
                self.output[i] = next(self._generator)
        except StopIteration:
            self.finished <<= True
