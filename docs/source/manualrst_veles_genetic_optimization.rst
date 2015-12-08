=============================================
Automatic optimization of workflow parameters
=============================================

VELES can help to determine what parameters are the best for a particular model.
This is done using the
`genetic algorithm <https://en.wikipedia.org/wiki/Genetic_algorithm>`_-based
parameters optimization. The idea is to replace all the values in the configuration
file with classes inherited from :class:`veles.genetics.config.Tuneable`, such as
:class:`veles.genetics.config.Range`, and launch the workflow as usual, but
with ``--optimize`` option set.

``--optimize`` specifies the number of workflows evaluated within one generation
and optionally sets the number of generations to evaluate. By default, generations
keep emerging until there is any improvement. Here is a command line example::

   veles -s --optimize=50 <workflow> <config>

By default, the plotters are disabled in evaluated workflows. You can enabled them
by setting ``root.common.genetics.disable.plotting`` to ``False``::

   veles -s --optimize=50 <workflow> <config>- root.common.genetics.disable.plotting=False

.. note:: Make sure, that you run genetic with fixed random seed (`--random-seed`). Otherwise, the results with optimized configuration (after genetic) will be different.

::

    python3 -m veles -s --optimize=50 --random-seed 1234 <workflow> <config>

.. note:: For developers: do not change workflow, configuration file or any other settings while workflow in optimization mode is running. Optimization loads code each time when another chromosome is running. So, better to copy or backup your Project before run in optimization mode. Otherwise, the results with optimized configuration (after genetic) could be different.

::

    cp -r Veles VelesGenetic
    cd VelesGenetic
    python3 -m veles -s --optimize=50 --random-seed 1234 <workflow> <config>

By default, there will appear a plotter which shows the progress in terms of the
selected metric. The metric value is got internally from any unit which
implement :class:`veles.result_provider.IResultProvider` interface and specify
"EvaluationResult" in :func:`get_metric_values()`. Such units include :class:`veles.znicz.decision.DecisionGD` and
:class:`veles.znicz.decision.DecisionMSE`.

By default, only validation metric value is optimized. If it
is needed to take train into account as well, or to do some non-linear processing of the
metric value, one can override ``root.common.evaluation_transform`` function.
For example, if we want to optimize the sum of validation and train metrics::

   veles -s --optimize=50 <workflow> <config> "root.common.evaluation_transform=lambda v, t: v + t"

Internally, Veles launches an instance of
:class:`veles.genetics.optimization_workflow.OptimizationWorkflow`. It supports
distributed operation, so you can parallelize the models' evaluation as usual.

After optimization you will see something like this::

    INFO:GeneticsOptimizer:Best fitness: 0.98
    INFO:GeneticsOptimizer:Best snapshot:
    INFO:GeneticsOptimizer:Best configuration
    --------------------------------------------------------------------------------
    Configuration "root.mnistr":
    {'layers': [{'->': {'bias_filling': 'uniform',
                        'bias_stddev': 0.026520000000000005,
                        'output_sample_shape': 220,
     ...
     }
    --------------------------------------------------------------------------------
    INFO:GeneticsOptimizer:Best configuration was saved to MNIST.mnist_best_config.py

Copy best configuration parameters to the <config> file from log or find best_config.py file in the folder, where Genetic optimization was running. Run workflow with the best configuration and with the same random seed to check the results::

    python3 -m veles -s --random-seed 1234 <workflow> <best_config>

