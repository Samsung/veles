:orphan:

Using Configs
:::::::::::::

Class :class:`veles.config.Config` is a versatile configuration storage,
suitable for managing Model's (:doc:`manualrst_veles_models`) parameters.
Predefined parameters are located at :doc:`manualrst_veles_workflow_parameters`.

1. Import the global configuration root node

   .. code-block:: python

      from veles.config import root

   Also, custom configuration object can be created:

   .. code-block:: python

      from veles.config import Config

      my_config = Config()

   The assignment operator can be used to set the values of configuration objects

   .. code-block:: python

      root.my_workflow.loader.minibatch_size = 100

   The code above creates the object ''root.my_workflow.loader.minibatch_size''
   and set it to 100.

   .. code-block:: python

      my_config = Config()
      my_config.my_workflow.decision.fail_iteration = 20

2. Besides, :meth:`update() <veles.config.Config>` can be used to setting
arguments in the form of a tree.

   .. code-block:: python

      root.my_workflow.update({
          "snapshotter": {"prefix": "my_workflow"},
          "loader": {"minibatch_size": 88, "on_device": True},
          "layers": [364, 10]})

   It is a more convenient form which is equivalent to:

   .. code-block:: python

      root.my_workflow.snapshotter.prefix = "my_workflow"
      root.my_workflow.loader.minibatch_size = 88
      root.my_workflow.loader.on_device = True
      root.my_workflow.layers = [364, 10]

   .. note:: :class:`dict` values can not be assigned with :meth:`update() <veles.config.Config>`. Use the assignment operator instead:

   .. code-block:: python

      root.my_workflow.parameter = {"key1": "value1", "key2": "value2"}

3. ``root.common`` is the general Veles configuration node applyed on the level
of Workflow (just a convention). Common arguments
are located at :mod:`veles.config` or :doc:`manualrst_veles_common_parameters`

   .. code-block:: python

      root.common.update({"precision_type": "float",
                          "precision_level": 0})

4. Parameters are set in workflow file, configuration file and in the
command line.

  Arguments application order is:

  1. Workflow. Workflow has default parameters. Default parameters can not be
  deleted, but it can be moved to the first configuration file. If workflow
  hasn't default parameters, then it is defined in the configuration file.

  In my_workflow.py:
  .. code-block:: python

     root.my_workflow.loader.minibatch_size = 40

  2. Configuration files. Arguments in configuration file (my_config.py)
  update parameters in the workflow file (my_workflow.py). Model can be executed
  with many configuration files. They will update each other in the order in
  which they appear on the command line::

     veles my_workflow.py my_config1.py my_config2.py

  In my_workflow.py:

  .. code-block:: python

     root.my_workflow.loader.minibatch_size = 40

  In my_config1.py:

  .. code-block:: python

     root.my_workflow.loader.minibatch_size = 88

  In my_config2.py:

  .. code-block:: python

     root.my_workflow.loader.minibatch_size = 30

  .. image:: _static/configs.png

The resulting minibatch_size will be 30.

  3. Command line. Parameters in the command line overwrite arguments in
  configuration file (my_config.py). The parameters can be defined after
  workflow and configuration files on the command line::

     veles my_workflow.py my_config1.py my_config2.py root.my_workflow.loader.minibatch_size=20 root.common.disable_plotting=True

  Result minibatch_size will be 20.

5. Arguments can be used after setting the value of configuration objects.
For example:

  .. code-block:: python

      from veles.config import root

      root.my_workflow.update({
          "decision": {"fail_iterations": 20,
                       "max_epochs": 300})

      print("Fail iterations is ", root.my_workflow.decision.fail_iterations)

6. Here is an example of using Loader's configuration parameters in the
workflow.

  Data parameters:

  .. code-block:: python

      root.my_workflow.update({
           ...

           "loader": {"minibatch_size": 40,
                      "filename_types": ["jpeg"],
                      "color_space": "RGB",
                      "train_paths": "/home/Desktop/MyData"]},
           ...
       })


  Example of Loader with this parameters (Note that we strongly discourage to
  use configuration parameters in the Unit code!! It should be done on the level of the Workflow):

  .. code-block:: python

      self.loader = Loader(
          self,
          minibatch_size=root.my_workflow.loader.minibatch_size,
          filename_types=root.my_workflow.loader.filename_types
          color_space=root.my_workflow.loader.color_space
          train_paths=root.my_workflow.loader.train_paths)


  Parameters can be setted like this if all configuration objects names exactly
  match with kwargs of the Loader class:

  .. code-block:: python

      self.loader = Loader(
          self, **root.my_workflow.loader.__content__)

  If :class:`veles.znicz.standard_workflow.StandardWorkflow` was used for
  creating workflow, loader's parameters should be passed as configuration
  objects to :func:`veles.__main__.Main._load()`

  .. code-block:: python

      from veles.znicz.standard_workflow import StandardWorkflow


      def run(load, main)
          load(StandardWorkflow,
               loader_config=root.my_workflow.loader)
          main()
