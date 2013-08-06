"""
Created on Aug 6, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from mpi4py import MPI
import threading
from mpi_peer import MPIPeer


class Master(MPIPeer):
    """
    Master of the main workflow which maintains links to nodes.
    """

    def __init__(self):
        """
        Constructor
        """
        self.shutting_down = False
        self.connections = {}
        self.thread_listener = threading.Thread(target=self.nodes_listener)
        self.thread_listener.start()

    def nodes_listener(self):
        port = MPI.Open_port()
        info = MPI.Info.Create()
        info.Set('ompi_global_scope', 'true')
        MPI.Publish_name(self.mpi_service_name, info, port)
        try:
            while not self.shutting_down:
                connection = MPI.COMM_WORLD.Accept(port, info)
                name = self.get_unique_connection_name()
                connection.send(name)
                self.connections[name] = connection
        finally:
            for connection in self.connections.values():
                connection.Free()
            MPI.Unpublish_name(self.mpi_service_name, info, port)
            MPI.Close_port(port)

    def get_unique_connection_name(self):
        return ""

    def shut_down(self):
        self.shutting_down = True
        self.thread_listener.join(1)
