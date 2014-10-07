============
Veles Basics
============

Veles starting point
::::::::::::::::::::
 |  Open in Eclipse Veles/scripts/velescli.py
 |  **Run->Debug As->Python Run**
 |  No error messages should appear only "usage: velescli.py ..." message

Run Veles example
:::::::::::::::::
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

Useful tools
::::::::::::

 |  Create dependences graph:
 |  ``sudo apt-get install graphviz``
 |  ``Veles/scripts/velescli.py --workflow-graph wf.png -d 0:0 -s Veles/veles/znicz/samples/mnist.py -``
 |  wait untill message **INFO:Workflow:Saving the workflow graph to wf.png** appears then stop process
 |  ``xdg-open wf.png``
