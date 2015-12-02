Distributed training. Command line examples
===========================================

Run workflow in distributed environment::

    # on master node
    python3 -m veles -l 0.0.0.0:5000 <workflow> <config>
    # on slave node
    python3 -m veles -m <master host name or IP>:5000 <workflow> <config>
    
.. note::
   5000 is the port number - use any you like!
 
Run workflow in distributed environment (known nodes)::

    # on master node
    python3 -m veles -l 0.0.0.0:5000 -n <slave 1>/cuda:0,<slave 2>/ocl:0:1 <workflow> <config>
 
.. note::   
   It's ok to use different backends to the each slave node. "ocl:0:1" sets the OpenCL device to use. "cuda:0" sets the CUDA device to use. Syntax can be much more complicated,
   for example, ``cuda:0-3x2`` launches 8 instances overall: two instances on each device
   from 0, 1, 2 and 3.