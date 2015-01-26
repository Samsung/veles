:orphan:

Using Configs
:::::::::::::

 Class :class:`veles.config.Config` was designed to set the Model (:doc:`manualrst_veles_models`) parameters.

In order to use it:

1. Import the root configuration node - instance ``root`` of :class:`veles.config.Config`::

    from veles.config import root

  Or you can create your own configuration object::

    from veles.config import Config

    my_config = Config()

  You can use "=" to set the value of configuration object::

    root.my_workflow.loader.minibatch_size = 100

  This line automatically creates the object ''root.my_workflow.loader.minibatch_size'' and assign to it a value 100. For your config::

    my_config = Config()
    my_config.my_workflow.decision.fail_iteration = 20

2. Also you can use :meth:`update() <veles.config.Config>`, which can set arguments in a form of tree::

    root.my_workflow.update({
        "snapshotter": {"prefix": "my_workflow"},
        "loader": {"minibatch_size": 88, "on_device": True},
        "layers": [364, 10]})

  It is a more convenient form which is equivalent to this::

    root.my_workflow.snapshotter.prefix = "my_workflow"
    root.my_workflow.loader.minibatch_size = 88
    root.my_workflow.loader.on_device = True
    root.my_workflow.layers = [364, 10]

3. ``root.common`` is a set of general parameters of Veles engine. You can see all common arguments in :class:`veles.config.Config` or :doc:`manualrst_veles_workflow_parameters`::

    root.common.update({"precision_type": "float",
                        "precision_level": 0}) # my_config.py

4. You can set parameters in workflow file, configuration file and in the command line.

  Arguments application order:

  1. Workflow. Workflow has default parameters. You can't delete default parameters, but you can move it to first configuration file::

       root.my_workflow.loader.minibatch_size = 40 # my_workflow.py

  2. Configuration files. Arguments in configuration file (my_config.py) update parameters in workflow file (my_workflow.py). You can run Model with many configuration files. They will update each other in the order in which they appear on the command line::

       veles my_workflow.py my_config1.py my_config2.py

     Arguments::

       root.my_workflow.loader.minibatch_size = 40 # my_workflow.py
       root.my_workflow.loader.minibatch_size = 88 # my_config1.py
       root.my_workflow.loader.minibatch_size = 30 # my_config2.py

   Result minibatch_size will be 30.

  3. Command line. Parameters in the command line overwrite arguments in configuration file (my_config.py). You can set the parameters after workflow and configuration files on command line::

       veles my_workflow.py my_config1.py my_config2.py root.my_workflow.loader.minibatch_size=20 root.common.plotters_disabled=True

   Result minibatch_size will be 20.

5. You can use arguments after setting value of configuration objects. For example::

    from veles.config import root

    root.my_workflow.update({
        "decision": {"fail_iterations": 20,
                     "max_epochs": 300})

    print("Fail iterations is ", root.my_workflow.decision.fail_iterations)

6. You can see all existing parameters in :doc:`manualrst_veles_workflow_parameters`.
