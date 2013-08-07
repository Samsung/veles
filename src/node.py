"""
Created on Aug 6, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import logging
from mpi4py import MPI
from src.mpi_peer import MPIPeer


class Node(MPIPeer):
    """
    Computational slave, that is, a separate process which actually calculates
    things.
    """

    def __init__(self):
        super(Node, self).__init__()
        self.shutting_down = False
        info = MPI.Info.Create()
        info.Set('ompi_global_scope', 'true')
        logging.debug("Lookup_name(%s)", self.mpi_service_name)
        port = MPI.Lookup_name(self.mpi_service_name, info)
        logging.debug("Connecting to %s...", str(port))
        self.connection = MPI.COMM_WORLD.Connect(port, info)
        logging.debug("Connected to %s...", str(port))
        self.id = self.connection.recv()
        logging.info("Connected to %s, id %s", str(port), self.id)

    def run_async(self):
        # TODO(v.markovtsev): launch a separate runner thread
        while not self.shutting_down:
            task = self.connection.recv()
            update = self.handle_task(task)
            self.connection.send(update)

    def shut_down(self):
        self.shutting_down = True
        # TODO(v.markovtsev): join with the runner thread
        self.connection.Free()

    def __fini__(self):
        self.shut_down()

    def handle_task(self, task):
        # TODO(v.markovtsev): do the work, producing the update object which
        # will be sent to master
        pass
