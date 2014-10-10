============
Veles Basics
============

Workflow consists of units. That's it.

.. _two_ways_of_running_veles:

Two ways of running Veles
^^^^^^^^^^^^^^^^^^^^^^^^^

You can run Veles in either of these two ways:

    * ``scripts/velescli.py ...``
    * ``python3 -m veles ...``
    
The second method executes ``veles/__main__.py``, which in turn calls
``scripts/velescli.py``, so they are totally equivalent.

Command line startup process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is the description of what happens after you execute velescli.py:

    #. A new instance of **Main** class is created and **run()** method is called.
       If **run()** returns, the resulting value is used as the return code for
       **sys.exit()**.
    #. "Special" command line arguments are checked for presence. They are listed in
       **Main.SPECIAL_OPTS** and include ``--help``, ``--html-help`` and ``--frontend``.
    #. 