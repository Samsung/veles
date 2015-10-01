=====================
Command line examples
=====================

View this HTML help::

    python3 -m veles --html-help
    
View help in terminal::

    python3 -m veles --help
    
Interactively compose the command line in web browser and run it::

    python3 -m veles --frontend
    
Run "Mnist" sample::

    python3 -m veles -s veles/znicz/samples/MNIST/mnist.py -
   
.. note:: 
   "-" is a shorthand for ``mnist_config.py``.
    
Draw specific workflow's scheme (replace ``workflow.py`` and ``config.py`` with
actual workflow and configuration paths)::

    python3 -m veles -p '' -s --disable-opencl --dry-run=init --workflow-graph scheme.png workflow.py config.py
    xdg-open scheme.png
    
Specify OpenCL device at once::

    python3 -m veles -d 0:0
    
.. note::
   The first number is the platform number and the second is the device number in that platform,
   for example "NVIDIA" platform and first (probably, only) device.

Run workflow without any OpenCL usage (slooow!)::

    python3 -m veles --disable-opencl

Disable plotters during workflow run::

    python3 -m veles -p ''
    
Do not send reports to web status server (for example, because it is not running)::

    python3 -m veles -s
    
Write only warnings and errors::

    python3 -m veles -v warning
    
Write extended information from unit of class "Class"::

    python3 -m veles --debug Class

Run workflow in distributed environment::

    # on master node
    python3 -m veles -l 0.0.0.0:5000 workflow.py config.py
    # on slave node
    python3 -m veles -m <master host name or IP>:5000 workflow.py config.py
    
.. note::
   5000 is the port number - use any you like!
 
Run workflow in distributed environment (known nodes)::
 
    # on master node
    python3 -m veles -l 0.0.0.0:5000 -n <slave 1>/0:0,<slave 2>/0:0 workflow.py config.py
 
.. note::   
   "0:0" sets the OpenCL device to use. Syntax can be much more complicated,
   for example, ``0:0-3x2`` launches 8 instances overall: two instances on each device
   from 0:0, 0:1, 0:2 and 0:3.