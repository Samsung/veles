# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 3, 2014.

ZeroMQ Twisted factory which is controlling ZeroMQ context.

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


from zmq import Context

from twisted.internet import reactor


class ZmqContextManager(object):
    """
    I control individual ZeroMQ connections.

    Factory creates and destroys ZeroMQ context.

    :var reactor: reference to Twisted reactor used by all the connections
    :var ioThreads: number of IO threads ZeroMQ will be using for this context
    :vartype ioThreads: int
    :var lingerPeriod: number of milliseconds to block when closing socket
        (terminating context), when there are some messages pending to be sent
    :vartype lingerPeriod: int

    :var connections: set of instantiated :class:`ZmqConnection`
    :vartype connections: set
    :var context: ZeroMQ context
    """

    reactor = reactor
    ioThreads = 1
    lingerPeriod = 100
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ZmqContextManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        """
        Constructor.

        Create ZeroMQ context.
        """
        if not self.initialized:
            self.initialized = True
            self.connections = set()
            self.context = Context(self.ioThreads)
            reactor.addSystemEventTrigger('during', 'shutdown', self.shutdown)

    def __repr__(self):
        return "ZmqContextManager(%d threads)" % self.ioThreads

    def shutdown(self):
        """
        Shutdown factory.

        This is shutting down all created connections
        and terminating ZeroMQ context. Also cleans up
        Twisted reactor.
        """
        if not self.initialized:
            return
        self.initialized = False
        for connection in self.connections.copy():
            connection.shutdown()

        self.connections = None
        self.context.term()
        self.context = None
