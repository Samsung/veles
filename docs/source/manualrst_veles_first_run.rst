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

    Without plotters:
    Veles/scripts/velescli.py -d 0:0 -s Veles/veles/znicz/samples/mnist.py - -p ''

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

========================
Periodical system update
========================

In case of new packages and modules release perform periodical system update with following command

``sudo apt-get update && sudo apt-get upgrade``

``sudo pip3 install -r requirements.txt --upgrade``

=========================================================
Installing Sphinx (python system documentation generator)
=========================================================
Go to the `Veles/docs` folder.

Make sure that python2 version of Sphinx is NOT installed.

``sudo apt-get purge sphinx*``

Then install `python3` Sphinx-doc:

``apt-get install sphinx3-doc``

``sudo pip3 install -r Veles/docs/requirements.txt``

Then try to generate documentation:

``python3 generate_docs.py``

Documentation could be located in Veles/docs/build/html/index.html

=================================
Nvidia Driver & Cuda installation
=================================

Download from: /data/veles/Drivers

Change to tty1 terminal (CTRL+ALT+F1)

``sudo service lightdm stop``

NVIDIA-Linux..run --no-opengl-files

``sudo service lightdm start``

Cuda installation: should be straightforward, pay attention to **not install driver during cuda installation process**

====================
Nvidia Drivers issue
====================

In case of drivers problem check that there is no conflicts with `nouveau` driver:

``sudo modprobe nvidia``

If error appears (modprobe: ERROR: could not insert 'nvidia': No such device) check `nouveau` driver presence:

``lsmod | grep nouveau``

If nouveau found add it to the black list:

``sudo nano /etc/modprobe.d/blacklist-framebuffer.conf``

Add `blacklist nouveau`

Reinstall nvidia driver