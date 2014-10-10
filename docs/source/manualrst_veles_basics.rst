======
Basics
======

Workflow consists of units. That's it.

.. _two_ways_of_running_veles:

Two ways of running Veles
:::::::::::::::::::::::::

You can run Veles in either of these two ways:

    * ``scripts/velescli.py ...``
    * ``python3 -m veles ...``
    
The second method executes :mod:`veles.__main__`, which in turn calls
``scripts/velescli.py``, so they are totally equivalent.

.. include:: manualrst_veles_units.rst


    
       
.. include:: manualrst_veles_launcher.rst