"""
Created on Feb 10, 2014

Workflow launcher (server/client/standalone).

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import argparse
import paramiko
import socket

import client
import config
import server
import units
import web_status


class Launcher(units.Unit):
    """Workflow launcher.
    """
    def __init__(self, workflow, **kwargs):
        super(Launcher, self).__init__(workflow, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--server_address", type=str, default="",
            help="Workflow will be launched in client mode "
            "and connected to the server at the specified address.")
        parser.add_argument("-l", "--listen_address", type=str, default="",
            help="Workflow will be launched in server mode "
            "and will accept client connections at the specified address.")
        args = parser.parse_args()
        if len(args.server_address):
            self.agent = client.Client(args.server_address, workflow)
        elif len(args.listen_address):
            self.agent = server.Server(args.server_address, workflow)
        else:
            self.agent = workflow
        self.web_status = None
        # Launch the status server if it's not been running yet
        self.launch_status(True)

    def initialize(self, **kwargs):
        return self.agent.initialize(**kwargs)

    def run(self):
        return self.agent.run()

    def stop(self):
        if self.web_status:
            self.web_status.stop()
        self.agent.stop()

    def launch_status(self, check=False):
        if check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", config.web_status_port))
        if not check or result == 0:
            self.log().info("Launching the web status server")
            self.web_status = web_status.WebStatus(self.agent)
            self.web_status.run()

    @staticmethod
    def launch_node(node, script, server_address=socket.gethostname()):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(node, look_for_keys=True)
        client.exec_command("nohup %s --server_address %s" % (
                                        script, server_address))
        client.close()
