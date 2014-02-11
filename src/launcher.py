"""
Created on Feb 10, 2014

Workflow launcher (server/client/standalone).

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import server
import client
import argparse
import socket
import paramiko


class Launcher(units.Unit):
    """Workflow launcher.
    """
    def __init__(self, workflow, **kwargs):
        super(Launcher, self).__init__(workflow, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument("-server_address", type=str, default="",
            help="Workflow will be launched in client mode "
            "and connected to the server at the specified address.")
        parser.add_argument("-listen_address", type=str, default="",
            help="Workflow will be launched in server mode "
            "and will accept client connections at the specified address.")
        args = parser.parse_args()
        if len(args.server_address):
            self.factory = client.Client(args.server_address, workflow)
        elif len(args.listen_address):
            self.factory = server.Server(args.server_address, workflow)
        else:
            self.factory = workflow

    def initialize(self, **kwargs):
        return self.factory.initialize(**kwargs)

    def run(self):
        return self.factory.run()

    @staticmethod
    def launch_node(node, script, server_address=socket.gethostname()):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(node, look_for_keys=True)
        client.exec_command("nohup %s --server_address %s" % (
                                        script, server_address))
        client.close()
