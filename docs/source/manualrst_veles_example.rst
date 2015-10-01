===============================
How to use Veles. MNIST example
===============================

:::::::::::::
Preprocessing
:::::::::::::

For preprocessing of the data you should use Loaders (descendants of
:class:`veles.loader.base.Loader`). To speed up process of training, create one Workflow for
preprocessing and one Workflow for training. Preprocessing workflow could look
like this:

.. image:: _static/preprocessing_workflow.png

Here Loader preprocesses the data and DataSaver saves it. If speed is
not an issue, you can use preprocessing inside of training Workflow.

Avaliable Loaders can be found in :doc:`manualrst_veles_units_kwargs`. Some of
Loaders could be used right away.
For other Loaders some functions has to be defined in descendant Class.

Sometimes existing Loaders are not suitable for the specific task. In such case
you should write custom Loader. For example, Loader for MNIST dataset
(:class:`veles.znicz.loader.loader_mnist.MnistLoader`) looks like this:

.. code-block:: python

    @implementer(IFullBatchLoader)
    class MnistLoader(FullBatchLoader):
        MAPPING = "mnist_loader"
        ...

        def __init__(self, workflow, **kwargs):
            super(MnistLoader, self).__init__(workflow, **kwargs)
            ...

        def load_dataset(self):
            """
            Loads dataset from internet
            """
            ...

        def load_original(self, offs, labels_count, labels_fnme, images_fnme):
            """Loads data from original MNIST files (idx1-ubyte and idx3-ubyte)
            """
            ...

        def load_data(self):
            """Here we will load MNIST data.
            """
            ...
            self.load_dataset()
            self.load_original(...)
            ...

Each Loader, which can be used without deriving from it, must have
MAPPING - unique loader's name.

There are few types of Loaders. If Loader was derived from
:class:`veles.loader.fullbatch.FullBatchLoader`, data will be stored
entire in memory. If Loader was derived from :class:`veles.loader.base.LoaderMSE`,
Loader has not only labels (any string, int or double values),
but also targets (matrixes, vectors). If Loader was derived from
:class:`veles.loader.image.ImageLoader`, functionality for preprocessing images
could be used (scale, crop, add sobel, rotate, change color space, mirror, etc.)

Any descendants of :class:`veles.loader.fullbatch.FullBatchLoader` must implement
IFullBatchLoader interface :class:`veles.loader.fullbatch.IFullBatchLoader`:

.. code-block:: python

    class IFullBatchLoader(Interface):
        def load_data():
            """Load the data here.
            Must be set: class_lengths, original_data, [original_labels].
            """

In load_data() you should define:

  1. `class_lengths` (size of train, validation and test samples),

  2. `original_data` (instance of :class:`veles.memory.Array` with array of data [test samples...validation samples...train samples]),

  3. `original_labels` (list of labels [test labels...validation labels...train labels]).
Lenghts of `original_data` and `original_labels` must be equal.

If you have only data for train, you should use
:func:`veles.loader.fullbatch._resize_validation` to extract validation set from train set
(percentage of train set to validation defines by `validation_ratio` parameter)

For any Loader normalization can be set. Avalible types of normalization can be
found in :doc:`manualrst_veles_workflow_parameters` in Data parameters.

Any Loader shuffles each train minibatch by default.
(To change it use `shuffle_limit`)

Loader prints simple statistics about data and compares labels
distribution in train, validation and test sets.


::::::::
Training
::::::::

There are 4 ways to create train Workflow:

1. Use existing Snapshot of Workflow and continue training.
2. Use existing Workflow and existing Configuration file.
3. Use existing Workflow and change Configuration file.
4. Create custom Workflow and Configuration file.

+++++++++++++++++++++
Use existing Snapshot
+++++++++++++++++++++

Use your Snapshot to continue training if training was interrupted.

Or download existing Snapshot from Amazon (pathes to snapshots
of Model is in "snapshots" field in manifest.json)

Here is manifest.json of MNIST:

.. code-block:: python

    {
    ...
    "snapshots":
    ["https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_validation_1.92_train_0.04.4.pickle.gz",
    "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_caffe_validation_0.86_train_0.23.4.pickle",
    "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_validation_0.73_train_0.11.4.pickle"]
    }

Or use path to amazon snapshot as command line argument. For fully-connected MNISTWorkflow::

    python3 -m veles -s -d 0 -w=https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_validation_1.92_train_0.04.4.pickle.gz veles/znicz/samples/MNIST/mnist.py -

For convolutional MNISTWorkflow::

    python3 -m veles -s -d 0 -w=https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_validation_0.73_train_0.11.4.pickle veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_conv_config.py

Trained snapshot from Amazon has best accuracy, so it is pointless to train
already trained snapshot. Better use it for testing or train the ensemble.

+++++++++++++++++++++++++++++++++++++++++++++++++++++
Use existing Workflow and existing Configuration file
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Use existing Workflows from samples or download them from VelesForge.

To run veles from command line you need to set path to workflow and path to
configuration files as arguments::

    python3 -m veles path_to_workflow path_to_config

See :doc:`manualrst_veles_cml_examples` for command line examples.
To run MNIST workflow from command line for fully-connected Workflow::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py -

For convolutional Workflow::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_conv_config.py

For convolutional Workflow with Caffe configuration::

    python3 -m veles -s -d 0 veles/znicz/samples/MNIST/mnist.py veles/znicz/samples/MNIST/mnist_caffe_config.py

If veles was installed for 1-2 users levels, set
PYTHONPATH="/usr/lib/python3/dist-packages" or use absolute paths to the Workflow
and Configuration files::

    python3 -m veles -s -d 0 /usr/lib/python3/dist-packages/veles/znicz/samples/MNIST/mnist.py -

Or copy samples from /usr/lib/python3/dist-packages/veles/znicz/samples to your local directory.

+++++++++++++++++++++++++++++++++++++++++++++++++++
Use existing Workflow and change Configuration file
+++++++++++++++++++++++++++++++++++++++++++++++++++

About configuration: :doc:`manualrst_veles_configuration`

First copy samples to local directory::

    cp -r /usr/lib/python3/dist-packages/veles/znicz/samples /home/user/

or download Model from VelesForge.

There 2 ways to change configuration parameters:

1. In configuration file
2. From command line

To use first way, open file /home/user/samples/MNIST/mnist_config.py and change it.
To use second way change necessary parameters right from command line::

    python3 -m veles -s -d 0 /usr/lib/python3/dist-packages/veles/znicz/samples/MNIST/mnist.py - root.mnistr.loader.minibatch_size=10 root.mnistr.loader.data_path=\"/path/to/new/dataset\"

MNIST workflow (:class:`veles.znicz.samples.mnist.MnistWorkflow`) was derived
from StandardWorkflow (:class:`veles.znicz.standard_workflow.StandardWorkflow`).

See parameters of StandardWorkflow and how to work with it here: :doc:`manualrst_veles_workflow_parameters`

To change loss function from Softmax to MSE, change `loss_function` parameter.
Don't forget to change last layer in `layers` from "softmax" type to "all2all" type.

.. code-block:: python

    ...
    root.mnistr.update({
        ...
        "loss_function": "mse", # use to be softmax
        ...
        "layers": [{...},
                   {"name": "fc_softmax2",
                    "type": "all2all", # use to be softmax
                    "->": {...},
                    "<-": {...}}]})


.. note:: Name of layer in `layers` parameter does not define type of layer.
   Layer could have any name or could do not have name at all.

To customize loader change `loader_name` parameter. Make sure, that your
`loader_name` exists in MAPPING of some Loader and this Loader was imported
somewhere.

.. code-block:: python

    class MyLoader(SomeLoader):
        MAPPING = "my_loader"
        ...

.. code-block:: python

    from veles.znicz.loader.my_loader import MyLoader

    ...
    root.mnistr.update({
        ...
        "loader_name": "my_loader", # use to be mnist_loader
        ...})

To change parameters of preprocessing or loading data use `loader` parameters

.. code-block:: python

    ...
    root.mnistr.update({
        ...
        "loader": {"minibatch_size": Range(20, 1, 1000), # use to be Range(60, 1, 1000)
                   "force_numpy": False,
                   "normalization_type": "linear",
                   "data_path": "/path/to/new/dataset"}, # use to be os.path.join(root.common.dirs.datasets, "MNIST")
        ...})

If your Workflow failes to run because Loader was not initialized and some path
to data does not exist, make sure that dataset was downloaded (by Downloader
unit :class:`veles.downloader.Downloader` or manually), path to data exists and
has right permissions. Change data_path in `loader.data_path` if it is necessary.

To optimize parameters of Workflow by Genetic Algorithm use Range (:class:`veles.genetics.config.Range`)
for every parameter, which you want to optimize. When optimization is off, first
parameter will be used by default. In MNIST example minibatch size will be equal 20.
If optimization is on, second and third parameter will be used as range to optimize.
In MNIST example minibatch size will be selected from 1 to 1000 by Genetic Algorithm.

To change stop conditions of running process, use `decision` parameters.

.. code-block:: python

    ...
    root.mnistr.update({
        ...
        "decision": {"fail_iterations": 50,
                     "max_epochs": 1000000000},
        ...})

`fail iterations` parameter determines how much epochs without improvement in
validation accuracy should pass before training will be stopped

`max_epochs` parameter defines how much epochs should pass before training will be stopped

To change topology of Neural Network, use `layers` parameter.
Learn more: :doc:`manualrst_veles_workflow_parameters`

.. code-block:: python

    ...
    root.mnistr.update({
        ...
        "layers": [{"name": "fc_tanh1",
                    "type": "all2all_tanh",
                    "->": {"learning_rate": 0.1,
                           ...},
                    "<-": {...}},
                   {"name": "fc_softmax2",
                    "type": "softmax",
                    "->": {...},
                    "<-": {...}}]})

`layers` is a list of layers. The order of the list determines the order of layers.
Each layer has `type`, which defines unit's Class. `name` is optional parameter.
`"->"` defines forward propagation parameters. `"<-"` defines backward propagation parameters.

Other configuration parameters: for Snapshotter (descendants of :class:`veles.snapshotter.SnapshotterBase`)
use `snapshotter`, for LearningRateAdjuster (:class:`veles.znicz.lr_adjust.LearningRateAdjust`)
use `lr_adjuster`, for WeightsPlotter (:class:`veles.znicz.nn_plotting_units.Weights2D`)
use `weights_plotter`.

The above mentioned is valid only for StandardWorkflow
(:class:`veles.znicz.standard_workflow.StandardWorkflow`)

+++++++++++++++++++++++++++++++++++++++++++++
Create custom Workflow and Configuration file
+++++++++++++++++++++++++++++++++++++++++++++

To create a Workflow see :doc:`manualrst_veles_workflow_creation`.

.. code-block:: python

    ...
    class MnistWorkflow(StandardWorkflow):
        def __init__(self, workflow, **kwargs):
            super(MnistWorkflow, self).__init__(workflow, **kwargs)
            ...

        def link_mnist_weights_plotter(self, layers, limit, weights_input, parent):
            ...

        def create_workflow(self):
            ...

        def on_workflow_finished(self):
            ...

    def run(load, main):
        load(MnistWorkflow,
             ...)
        main()

:::::::
Testing
:::::::

::::::::::
How to run
::::::::::

:::::::::::::::::::::::
Optimization parameters
:::::::::::::::::::::::

:::::::::::::::
Export of Model
:::::::::::::::

::::::::::::::
Using plotters
::::::::::::::

:::::::::::::::
Using publisher
:::::::::::::::
