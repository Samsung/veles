"""
Created on Aug 6, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from mpi4py import MPI
from mpi_peer import MPIPeer


class Node(MPIPeer):
    """
    Computational slave, that is, a separate process which actually calculates
    things.
    """

    def __init__(self):
        info = MPI.Info.Create()
        info.Set('ompi_global_scope', 'true')
        port = MPI.Lookup_name('veles', info)
        self.connection = MPI.COMM_WORLD.Connect(port, info)
        self.id = self.connection.recv()

    def __fini__(self):
            self.connection.Free()
