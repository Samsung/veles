=================
Distributed setup
=================

Same paths
::::::::::

Veles supposes that each slave **and master** have access to the same root path,
in other words, Veles must be accessible by the same path on any node. Will it be
NFS or Ceph or Lustre or anything else - it does not matter, except for speed.

SSH access
::::::::::

If you are not going to use Hadoop and plan to use pure Veles, then you will
have to setup passwordless SSH access between master node and slave nodes.
Normally, executing on master node:

::

    ssh-copy-id <any slave>
    
is enough.

Logging setup
:::::::::::::

To enable centralized logs collection, you must install a MongoDB server.
On Ubuntu, ``sudo apt-get install mongodb-org`` is enough, provided by addition
of the official MongoDB repository as described in `the manual <http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/#install-mongodb>`_.

The server address is passed via ``--log-mongo`` or just ``-g`` option and must be
of the form "<host or IP>:port", e.g. "192.168.0.2:3000".

All the neccessary tables (veles.logs and veles.events), as well as indices are
created automatically during the start of ``velescli.py`` and web status server.

:doc:`manualrst_veles_mongo`
