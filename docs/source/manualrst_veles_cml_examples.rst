=====================
Command line examples
=====================

View this HTML help::

    python3 -m veles --html-help
    
View help in terminal::

    python3 -m veles --help
    
Interactively compose the command line in web browser and run it::

    python3 -m veles --frontend
    
To run the Model, set path to the workflow and path to the configuration file (replace ``<workflow>`` and ``<config>`` with
actual workflow and configuration paths)::

    python3 -m veles <workflow> <config>

Run "Mnist" sample::

    python3 -m veles -s veles/znicz/samples/MNIST/mnist.py -

.. note::
   "-" is a shorthand for ``veles/znicz/samples/MNIST/mnist_config.py``.

.. note::
   If you see warnings "Launcher:Failed to upload the status", use "-s" option to disable reporting status to the Web Status Server.
    
Specify OpenCL device at once::

    python3 -m veles -d 0:0 <workflow> <config>

.. note::
   The first number is the platform number and the second is the device number in that platform,
   for example "NVIDIA" platform and first (probably, only) device.

Specify CUDA device at once::

    python3 -m veles -d 0 <workflow> <config>

Change backend ("auto" for AutoDevice, "cuda" for CUDADevice, "numpy" for NumpyDevice, "ocl" for OpenCLDevice)::

    python3 -m veles --backend "cuda" <workflow> <config>

Training from snapshotted state of the model (replace ``<snapshot>`` with actual path to the snapshot, could be amazon link)::

    python3 -m veles --snapshot <snapshot> <workflow> <config>

Run in testing mode (use trained model)::

    python3 -m veles --test --snapshot <snapshot> <workflow> <config>

Run workflow without any OpenCL/CUDA/any accelerations  usage (slooow!)::

    python3 -m veles --force-numpy <workflow> <config>

Disable plotters during workflow run::

    python3 -m veles -p '' <workflow> <config>
    
Do not send reports to web status server (for example, because it is not running)::

    python3 -m veles -s <workflow> <config>
    
Write only warnings and errors::

    python3 -m veles -v warning <workflow> <config>
    
Write extended information from unit of class "Class"::

    python3 -m veles --debug Class <workflow> <config>

Draw specific workflow's scheme::

    python3 -m veles --workflow-graph scheme.png <workflow> <config>
    xdg-open scheme.png

.. include:: manualrst_veles_cml_examples_distributed.rst