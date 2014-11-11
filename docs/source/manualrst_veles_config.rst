:orphan:

Using Configs
:::::::::::::

 Class :class:`veles.config.Config` was designed to set the parameters of Model.

In order to use it:

1. Import most common usage - instance ``root`` of :class:`veles.config.Config`::

    from veles.config import root

  Or you can create your own config object::

    from veles.config import Config

    my_config = Config()

2. To set the value of config object you can use "="::

    root.loader.minibatch_size = 100

  This entry automatically creates an object ''root.loader.minibatch_size'' and assign to it a value 100. For your config::

    my_config = Config()
    my_config.mnist.decision.fail_iteration = 20

3. Also you can use :meth:`update() <veles.config.Config>` method::

    root.mnist.update({
        "all2all": {"weights_stddev": 0.05},
        "decision": {"fail_iterations": 300,
                     "snapshot_prefix": "mnist"},
        "loader": {"minibatch_size": 88, "on_device": True},
        "layers": [364, 10]})

  It is more convenient entry and it is equivalent to this::

    root.mnist.all2all.weights_stddev = 0.05
    root.mnist.decision.fail_iterations = 300
    root.mnist.decision.snapshot_prefix = "mnist"
    root.mnist.loader.minibatch_size = 88
    root.mnist.loader.on_device = True
    root.mnist.layers = [364, 10]

4. ``root.common`` is a common parameters. User can see all common parameters in :class:`veles.config.Config`, but changing should be in workflow (samples/mnist.py), configuration files (samples/mnist_config.py) or from command line::

    root.common.update({"precision_type": "float",
                        "precision_level": 0}) # mnist_config.py

5. You can set parameters in workflow file, configuration file and from command line.

  Parameters in configuration file (samples/mnist_config.py) update parameters in workflow file (samples/mnist.py). And parameters from command line update parameters from configuration file (samples/mnist_config.py)::

    root.mnist.loader.minibatch_size = 40 # mnist_config.py
    root.mnist.loader.minibatch_size = 88 # mnist.py

  minibatch_size will be 40. You can set the parameters from command line after workflow and configuration files::

    python3 -m veles -s veles/znicz/samples/mnist.py - root.mnist.minibatch_size=20 root.common.plotters_disabled=True

6. After setting value of config object, you can use it. For example::

    from veles.config import root
    from veles.znicz.decision import DecisionGD

    root.mnist.update({
        "decision": {"fail_iterations": 20,
                     "max_epochs": 300})

    class MnistWorkflow(nn_units.NNWorkflow):
        def __init__(self, workflow, layers, **kwargs):
            self.decision = DecisionGD(
                self, fail_iterations=root.mnist.decision.fail_iterations,
                max_epochs=root.mnist.decision.max_epochs)
