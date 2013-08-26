"""
Created on Aug 6, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import logging
from mpi4py import MPI
import threading
import time
from src.mpi_peer import MPIPeer


class TimeInterval(object):
    """
    The measure of the elapsed time of a task.
    """

    def __init__(self, start=time.time()):
        self.start = start
        self.finish = None

    def value(self):
        return self.finish - self.start


class Master(MPIPeer):
    """
    Master of the main workflow which maintains links to nodes.
    """

    def __init__(self):
        super(Master, self).__init__()
        self.shutting_down = False
        self.paused = False
        self.connections = {}
        self.updates_history = {}
        self.id_counter = 1
        self.thread_listener = threading.Thread(target=self.nodes_listener)
        self.thread_listener.start()

    def nodes_listener(self):
        port = MPI.Open_port()
        """Network address, encoded in the string (supplied by the system),
           at which the server will be able to accept connections from clients.
        """

        info = MPI.Info.Create()
        """Creates a new info object.
           The newly created object contains no key/value pairs.
        """
        info.Set('ompi_global_scope', 'true')

        logging.debug("Publishing %s...", self.mpi_service_name)
        MPI.Publish_name(self.mpi_service_name, info, port)
        """Publishes the pair (service_name, port_name) so that an application
           may retrieve port_name by calling MPI_Lookup_name with service_name
           as an argument.
           Requires ompi-server to be running.
        """
        logging.debug("Published %s", self.mpi_service_name)

        try:
            while not self.shutting_down:
                connection = MPI.COMM_WORLD.Accept(port, info)
                name = self.get_unique_connection_name()
                logging.debug("New node %s is connected", name)
                connection.send(name)
                self.init_node(name, connection)
                logging.info("New node %s has appeared online", name)
        finally:
            logging.debug("Tearing down the %d connections",
                          len(self.connections))
            for connection in self.connections.values():
                connection.Free()
            logging.debug("Unpublish_name()")
            MPI.Unpublish_name(self.mpi_service_name, info, port)
            logging.debug("Close_port()")
            MPI.Close_port(port)

    def get_unique_connection_name(self):
        res = str(self.id_counter)
        self.id_counter += 1
        return res

    def init_node(self, name, connection):
        self.connections[name] = connection
        self.updates_history[name] = []

    def shut_down(self):
        if self.shutting_down:
            return
        self.shutting_down = True
        logging.debug("Shutting down...")
        self.thread_listener.join(1)
        self.thread_listener = None
        logging.debug("Successfully joined with the listener thread")

    def __fini__(self):
        self.shut_down()

    def run(self):
        logging.debug("Starting the initial %d nodes...",
                      len(self.connections))
        for node_id in self.connections.keys():
            self.run_node(node_id)
        logging.debug("Entered the infinite loop")
        paused_nodes = self.connections.keys()
        while not self.shutting_down:
            updates = {}
            for node_id, conn in self.connections:
                update = conn.irecv()
                if update is not None:
                    logging.debug("Received an update from %s", node_id)
                    self.updates_history[node_id][-1].finish = time.time()
                    updates[node_id] = update
            for node_id, update in updates:
                self.handle_update(node_id, update)
                if not self.paused:
                    self.run_node(node_id)
                else:
                    paused_nodes.append(node_id)
            if not self.paused:
                for node_id in paused_nodes:
                    self.run_node(node_id)
                paused_nodes.clear()

    def handle_update(self, node_id, update):
        # TODO(v.markovtsev): parse the received object, apply node calculation
        # results
        pass

    def run_node(self, node_id):
        # TODO(v.markovtsev): construct the object that we will send async
        data_to_send = None
        logging.debug("Running %s", node_id)
        self.updates_history[node_id].append(TimeInterval())
        self.connections[node_id].isend(data_to_send)

    def pause(self):
        self.paused = True
