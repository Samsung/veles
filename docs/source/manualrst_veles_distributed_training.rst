==================================
Distributed Training: Master/slave
==================================


ZeroMQ-based star topology of distributed operation handles up to 100 nodes, depending on the task. Optionally, some compression can be applied. The state of things is sent to the Web Status Server.

The whole thing is fault tolerant, the death of a node does not cause the master stop. You can add new nodes dynamically, at any time, with any backends.

Snapshotting feature allows recovering from any disaster, restart the training in a different mode and with a different backend.

.. image:: _static/master-slave.png

Master process does not do any calculation and just serves other actors.
It stores the current workflow state, including all units' data. Slave
processes maintain two channels of communication with master: plain TCP (commands,
discovery, etc.) and ZeroMQ (data). Initially, a new slave connect to a TCP socket
on master, registers itself and starts sending job requests. Master receives job
requests, generates jobs (serialized data from each unit in the workflow) and sends them
to corresponding slaves. The thing worth noting is that **workflows that exists
in master and slave are the same**, they are just operated in different modes.

Master runs the graphics server (see :doc:`manualrst_veles_graphics`), so that any
number of client can draw plots of what's going on. Besides, it sends periodic
status information to the web status server via HTTP and listens to commands on
the same raw TCP socket which is used for talking to slaves. The special communication protocol
is used based on JSON.

.. include:: manualrst_veles_distributed_units.rst

.. include:: manualrst_veles_cml_examples_distributed.rst