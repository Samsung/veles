"""
Created on Jul 17, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import os
import platform
import subprocess


def show_file(file_name):
    system = platform.system()
    if system == "Windows":
        os.startfile(file_name)
    elif system == "Linux":
        subprocess.Popen(["xdg-open", file_name])
