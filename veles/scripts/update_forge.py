#!/usr/bin/env python3
"""
This script update all workflows to VelesForge

Command to run: FORGE_SERVER="http://smaug/forge" PYTHONPATH=`pwd`
veles/scripts/update_forge.py
"""


import subprocess
import os

from veles.config import root
from veles.scripts.generate_frontend import scan_workflows


def run(server_url):
    workflows = scan_workflows(False)
    for workflow in workflows:
        workflow_folder = os.path.dirname(workflow)
        if root.common.forge.manifest in os.listdir(workflow_folder):
            print("Update workflow %s in VELESForge" % workflow_folder)
            subprocess.call(
                ["python3", "-m", "veles", "forge", "upload", "-s", server_url,
                 "-d", workflow_folder])
        else:
            print("Workflow %s is not in Forge" % workflow)

if __name__ == "__main__":
    run(os.getenv("FORGE_SERVER"))
