# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jan 22, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import array
import binascii
import fcntl
import os
import six
import socket
import struct
from twisted.protocols.basic import LineReceiver
import uuid

from veles.logger import Logger


def interfaces():
    max_possible = 128
    max_bytes = max_possible * 32
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    names = array.array('B', b'\0' * max_bytes)
    outbytes = struct.unpack('iL', fcntl.ioctl(
        sock.fileno(),
        0x8912,  # SIOCGIFCONF
        struct.pack('iL', max_bytes, names.buffer_info()[0])
    ))[0]
    sock.close()
    if six.PY3:
        namestr = names.tobytes()
    else:
        namestr = names.tostring()
    for i in range(0, outbytes, 40):
        name = namestr[i:i + 16].split(b'\0', 1)[0]
        if name == b'lo':
            continue
        ip = namestr[i + 20:i + 24]
        yield (name.decode(), ip)


class NetworkAgent(Logger):
    """
    Stores the address and the port number.
    """

    CONFIG_ADDRESS = "address"
    CONFIG_PORT = "port"

    def __init__(self, configuration, workflow):
        """
        Parses the configuration file and loads CONFIG_ADDRESS and CONFIG_PORT
        """
        super(NetworkAgent, self).__init__()
        self._mid = None
        self._pid = None
        idx_semicolon = configuration.find(":")
        assert idx_semicolon >= 0
        self.address = configuration[:idx_semicolon]
        if not self.address:
            self.address = "0.0.0.0"
        self.port = int(configuration[idx_semicolon + 1:])
        self.debug("Network configuration: %s:%d", self.address, self.port)
        self._workflow = workflow
        self._launcher = workflow.workflow

    @property
    def pid(self):
        if self._pid is None:
            self._pid = os.getpid()
        return self._pid

    @property
    def mid(self):
        if self._mid is None:
            hwpart = ""
            dbusmidfn = "/var/lib/dbus/machine-id"
            if os.access(dbusmidfn, os.R_OK):
                with open(dbusmidfn) as midfd:
                    hwpart += midfd.read()[:-1]
            hwpart += "-%x" % uuid.getnode()
            chksum = 0
            for iname, iaddr in interfaces():
                chksum ^= struct.unpack("!L", iaddr)[0]
            swpart = binascii.hexlify(struct.pack("!L", chksum))
            self._mid = hwpart + "-" + swpart.decode('charmap')
            self.debug("My machine ID is %s", self._mid)
        return self._mid

    @property
    def workflow(self):
        return self._workflow

    @property
    def launcher(self):
        return self._launcher

    def initialize(self):
        self.workflow.thread_pool.start()


class StringLineReceiver(LineReceiver, object):
    def sendLine(self, line):
        if isinstance(line, str):
            if six.PY3:
                super(StringLineReceiver, self).sendLine(line.encode())
            else:
                LineReceiver.sendLine(self, line.encode())
        elif isinstance(line, bytes):
            if six.PY3:
                super(StringLineReceiver, self).sendLine(line)
            else:
                LineReceiver.sendLine(self, line)
        else:
            raise RuntimeError("Only str and bytes are allowed.")


class IDLogger(Logger):
    def __init__(self, logger=None, log_id=None):
        super(IDLogger, self).__init__(logger=logger)
        self.id = log_id

    def change_log_message(self, msg):
        return "%s: %s" % (self.id or "<none>", msg)
