Veles First Run
===============

=================
Python3.4 update
=================

Make shure you have only one instance of python3.x installed in system and it should be python3.4
 |  Run command ``python3``
 |  There should be sign "Python 3.4.0 (default, Apr 11 2014, 13:05:11)" or similar
 |  ``exit()``

Update your current packages:
(smaug repository should be added to the system: http://smaug.rnd.samsung.ru/apt)
::

    sudo apt-get clean
    sudo apt-get update
    sudo apt-get install --reinstall python3-twisted-experimental

Check that twisted is ready to use in python:
 |  ``python3``
 |  ``import twisted.web.client``
No error messages should appear!
 |  ``exit()``


=====================
Packages installation
=====================
::

    Install packages listed in Ubuntu.md (see Veles repo)
    Install python3 packages (sudo pip3 install pack_name) listed in requirements.txt

========================
PyDev interpreter change
========================

 |  Run **eclipse->Window->Preferences->PyDev->Interpreters->Python Interpreter**
 |  Add new interpreter with name python3.4 and path /usr/bin/python3.4
 |  Adjust builtins: **Forced Builtins->New:** twisted
 |  Save and close dialog window

=================================
Python project interpreter choose
=================================
 |  Import project in Eclipse: **File->Import->General->Existing project into Workspace**
 |  **Project Properties->PyDev-Interpreter->Interpreter**: choose python3.4

====================
Veles starting point
====================
 |  Open in Eclipse Veles/scripts/velescli.py
 |  **Run->Debug As->Python Run**
 |  No error messages should appear only "usage: velescli.py ..." message

=================
Run Veles example
=================
::

    Veles/scripts/velescli.py -s Veles/veles/znicz/samples/mnist.py -

    In case something went wrong:
    killall -9 velescli.py

    Only CPU usage (non OpenCL), useful in case of OpenCL problems (CPU only):
    Veles/scripts/velescli.py -s --cpu Veles/veles/znicz/samples/mnist.py -

    Select device:
    Veles/scripts/velescli.py -d 0:0 -s Veles/veles/znicz/samples/mnist.py -

============
Useful tools
============

 |  Check GPU temperature: ``nvidia-smi -l`` (looped)
 |
 |  Create dependences graph:
 |  ``sudo apt-get install graphviz``
 |  ``Veles/scripts/velescli.py --workflow-graph wf.png -d 0:0 -s Veles/veles/znicz/samples/mnist.py -``
 |  wait untill message **INFO:Workflow:Saving the workflow graph to wf.png** appears then stop process
 |  ``xdg-open wf.png``
