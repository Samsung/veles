"""
Created on Feb 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import paramiko


def launch_node(node, script):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(node, look_for_keys=True)
    client.exec_command("nohup " + script + " --slave")
    client.close()