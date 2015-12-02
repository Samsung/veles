=================================================================
How to use Veles. MNIST example. Simple (user entry/medium level)
=================================================================

:::::::::::::
Preprocessing
:::::::::::::

For preprocessing of the data you should use Loaders (descendants of
:class:`veles.loader.base.Loader`).

Available Loaders can be found in :doc:`manualrst_veles_units_kwargs`. Some of
Loaders could be used right away.

Each Loader, which can be used without deriving from it, must have
MAPPING - unique loader's name.

There are few types of Loaders. If Loader was derived from
:class:`veles.loader.fullbatch.FullBatchLoader`, data will be stored
entire in memory. If Loader was derived from :class:`veles.loader.base.LoaderMSE`,
Loader has not only labels (string, int or double values),
but also targets (matrixes, vectors). If Loader was derived from
:class:`veles.loader.image.ImageLoader`, functionality for preprocessing images
could be used (scale, crop, add sobel, rotate, change color space, mirror, etc.)

Learn more: :doc:`manualrst_veles_example_advanced`

::::::::
Training
::::::::

There are 4 ways to create train Workflow:

1. Use existing Snapshot of Workflow and continue training.
2. Use existing Workflow and existing Configuration file.
3. Use existing Workflow and change Configuration file.
4. Create custom Workflow and Configuration file. (Advanced)

+++++++++++++++++++++
Use existing Snapshot
+++++++++++++++++++++

Use your Snapshot to continue training if training was interrupted.

Or download existing Snapshot from Amazon (paths to snapshots
of the Model is in "snapshots" field in manifest.json)

Here is manifest.json of MNIST:

.. code-block:: python

    {
    ...
    "snapshots":
    ["https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_validation_1.92_train_0.04.4.pickle.gz",
    "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_caffe_validation_0.86_train_0.23.4.pickle",
    "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_validation_0.73_train_0.11.4.pickle"]
    }

Or use path to the amazon snapshot as command line argument. For fully-connected MNISTWorkflow::

    python3 -m veles -s -d 0 -w=https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_validation_1.92_train_0.04.4.pickle.gz veles/znicz/samples/MNIST/mnist.py -

For convolutional MNISTWorkflow::

    python3 -m veles -s -d 0 -w=https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_validation_0.73_train_0.11.4.pickle veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_conv_config.py

+++++++++++++++++++++++++++++++++++++++++++++++++++++
Use existing Workflow and existing Configuration file
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Use existing Workflows from samples or download them from :doc:`manualrst_veles_forge`.

To run veles from command line you need to set path to workflow and path to
configuration files as arguments::

    python3 -m veles path_to_workflow path_to_config

See :doc:`manualrst_veles_cml_examples` for command line examples.
To run the MNIST workflow from command line for fully-connected Workflow::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py -

For convolutional Workflow::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_conv_config.py

For convolutional Workflow with Caffe configuration::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_caffe_config.py

If Veles was installed for 1-2 users levels, set
PYTHONPATH="/usr/lib/python3/dist-packages" or use absolute paths to the Workflow
and Configuration files::

    python3 -m veles -s -d 0 /usr/lib/python3/dist-packages/veles/znicz/samples/MNIST/mnist.py -

Or copy samples from /usr/lib/python3/dist-packages/veles/znicz/samples to your local directory.

+++++++++++++++++++++++++++++++++++++++++++++++++++
Use existing Workflow and change Configuration file
+++++++++++++++++++++++++++++++++++++++++++++++++++

About configuration: :doc:`manualrst_veles_configuration`

First copy samples to a local directory::

    cp -r /usr/lib/python3/dist-packages/veles/znicz/samples /home/user/

or download Model from :doc:`manualrst_veles_forge`.

There 2 ways to change configuration parameters:

1. In configuration file
2. From command line

To use the first way, open file /home/user/samples/MNIST/mnist_config.py and change it.
To use the second way change necessary parameters right from the command line::

    python3 -m veles -s -d 0 /usr/lib/python3/dist-packages/veles/znicz/samples/MNIST/mnist.py - root.mnistr.loader.minibatch_size=10 root.mnistr.loader.data_path=\"/path/to/new/dataset\"

MNIST workflow (:class:`veles.znicz.samples.mnist.MnistWorkflow`) was derived
from StandardWorkflow (:class:`veles.znicz.standard_workflow.StandardWorkflow`).

See parameters of StandardWorkflow and how to work with it here: :doc:`manualrst_veles_workflow_parameters`

Learn more: :doc:`manualrst_veles_example_advanced`

:::::::
Testing
:::::::

Snapshot of trained Workflow is required to run Workflow in testing mode.
There are 5 ways to create test Workflow:

1. Use existing test Workflow or script
2. Use --test and --result-file arguments in command line
3. Use testing mode and write_results function (Advanced)
4. Create Workflow with extract_forward function (Advanced)
5. Create custom test Workflow (Advanced)

++++++++++++++++++++++++++++++++++++
Use existing test Workflow or script
++++++++++++++++++++++++++++++++++++

If test Workflow is exists it is located in the directory with
train Workflow. See samples or download Model from :doc:`manualrst_veles_forge`.

++++++++++++++++++++++++++++++++++++++++++++++++++++++
Use --test and --result-file arguments in command line
++++++++++++++++++++++++++++++++++++++++++++++++++++++

If Loader of trained Workflow has filled test set, run
Workflow in testing mode with --test command line argument.
Use --result-file argument to save the results of testing::

    python3 -m veles -s -d 0 --test --result-file="/home/user/mnist_result.txt" /home/user/samples/MNIST/mnist.py -

File with results will be constructed from results of
``get_metric_values`` and ``get_metric_names`` functions of Units
(IResultProvider must be implemented). Example:

.. code-block:: python

    @implementer(IResultProvider, ...)
    class EvaluatorBase(...):
        ...
        def get_metric_names(self):
            ...

        def get_metric_values(self):
            ...


Learn more: :doc:`manualrst_veles_example_advanced`

::::::::::
How to run
::::::::::

+++++++++++++++++++++++++
Run with ipython notebook
+++++++++++++++++++++++++

Veles is usable from IPython or IPython Notebook.
Open ipython notebook, import veles and run it:

.. code-block:: python

    import veles
    launcher = veles(
        "veles/znicz/samples/MnistSimple/mnist.py", stealth=True,
        matplotlib_backend="WebAgg")

Arguments are the same as for the command line, but "-" symbol changes to "_" symbol
and using of long form options is required.

To pause the process of execution:

.. code-block:: python

    launcher.pause()

To resume the process of execution:

.. code-block:: python

    launcher.resume()

To stop the process of execution:

.. code-block:: python

    launcher.stop()

To initialize the Workflow:

.. code-block:: python

    launcher.initialize()

To run the Workflow:

.. code-block:: python

    launcher.run()

To initialize and run the Workflow:

.. code-block:: python

    launcher.boot()

To get the Workflow:

.. code-block:: python

    launcher.workflow

To get Units:

.. code-block:: python

    launcher.workflow.units

To get specific Unit:

1. Get by name:

.. code-block:: python

    loader = launcher.workflow["MnistLoader"]

2. Get by the instance of Unit:

.. code-block:: python

    loader = launcher.workflow.loader

3. Get from Units list

.. code-block:: python

    launcher.workflow.units

[veles.plumbing.StartPoint "Start of MnistWorkflow",
veles.plumbing.EndPoint "End of MnistWorkflow",
<veles.plumbing.Repeater object at 0x7f8fc4f1def0>,
<MnistSimple.loader_mnist.MnistLoader object at 0x7f8ff17c20f0>,
...]

.. code-block:: python

    loader = launcher.workflow.units[3]

+++++++++++++++++++++
Run from command line
+++++++++++++++++++++

See :doc:`manualrst_veles_cml_examples`.

+++++++++++++++
Frontend option
+++++++++++++++

Use ``frontend`` option for the interactive display of Veles options and the command line.
Run in the terminal::

    python3 -m veles --frontend

Compose the command line and click run button.

.. image:: _static/web_frontend.png

++++++++++++++
Manhole option
++++++++++++++

Use manhole option to run interactive mode at any time. Run::

    python3 -m veles --manhole /home/user/samples/MNIST/mnist.py -

You will see something like this::

    MANHOLE:Manhole UDS path: nc -U /tmp/manhole-7355
    MANHOLE:Waiting for a new connection (in pid 7355) ...

To switch to the interactive console open new terminal and run the command with a Manhole UDS path::

    nc -U /tmp/manhole-7355

You will see::

    VELES interactive console
    Type in 'workflow' or 'units' to start
    veles [1]>

Change some attributes. For example, decrease learning rate in backward propagation units (gds) in 10 times::

    veles [1]> for gd in workflow.gds:
          ...:     gd.learning_rate/=10
          ...:

    veles [2]>

To stop interactive mode and continue execution type "exit()"::

    veles [2]> exit()


++++++++++++++++++++
Distributed training
++++++++++++++++++++

See :doc:`manualrst_veles_distributed_training`.


++++++++++++++++++
Training ensembles
++++++++++++++++++

See :doc:`manualrst_veles_ensembles`.

:::::::::::::::::::::::
Optimization parameters
:::::::::::::::::::::::

.. code-block:: python

    from veles.config import root
    from veles.genetics import Range

    root.mnistr.update({
        ...
        "loader": {"minibatch_size": Range(20, 1, 1000),
                   "normalization_type": "linear",
                   "data_path": "/path/to/dataset"},
        ...})


To optimize parameters of Workflow by Genetic Algorithm use Range
(veles.genetics.config.Range) for every parameter, which you want to optimize.
When optimization is off, the first parameter will be used by default. In MNIST
example minibatch size will be equal 20. If optimization is on, the second and
the third parameter will be used as range to optimize. In MNIST example
minibatch size will be selected from 1 to 1000 by Genetic Algorithm.

See :doc:`manualrst_veles_genetic_optimization`.

:::::::::::::::::::
Export of the Model
:::::::::::::::::::

To export Model as package use :func:`veles.workflow.package_export`.
Set path to the exported package by `package_name` argument. `precision` is an optional parameter.

.. code-block:: python

    ...
    class MnistWorkflow(StandardWorkflow):
        def __init__(self, workflow, **kwargs):
            super(MnistWorkflow, self).__init__(workflow, **kwargs)
            self.export_wf = kwargs.get("export_wf", False)
            self.package_name = kwargs.get(
                "package_name", os.path.join(root.common.dirs.user, "mnist.zip"))

        ...

        def on_workflow_finished(self):
            super(MnistWorkflow, self).on_workflow_finished()
            if self.export_wf:
                self.package_export(self.package_name, precision=16)

    def run(load, main):
        load(MnistWorkflow,
             ...)
        main()


::::::::::::::
Using plotters
::::::::::::::

To disable plotters during Workflow run::

    python3 -m veles -p '' /home/user/samples/MNIST/mnist.py -

To choose WebAgg backend::

    python3 -m veles -p 'WebAgg' /home/user/samples/MNIST/mnist.py -

To choose Qt4Agg backend::

    python3 -m veles -p 'Qt4Agg' /home/user/samples/MNIST/mnist.py -

To disable plotting service::

    python3 -m veles /home/user/samples/MNIST/mnist.py - root.common.disable.plotting=True

:::::::::::::::
Using publisher
:::::::::::::::

See :doc:`manualrst_veles_publishing`.
