:orphan:

Parameters of the StandardWorkflow
::::::::::::::::::::::::::::::::::

The class :class:`veles.config.Config` is a hierarchical dictionary designed to pass
complex parameters into Unit instances during initialization without prior knowledge of
Unit requirement.

The Unit obviously has a knowledge of name of parameters used (dictionary) but
as class implementation does not work directly with config, because only Workflow
knows how many instances required for the ML model and what role they are playing
in the algorithm. Therefore Workflow passes config parameters to the Unit during
creation (instantiation).
To simplify the process the `root` (Config type) is declared as a global object
visible from any part of the program. It is also could be changed (overwritten)
from the command line. For detailed explanation, please see :doc:`manualrst_veles_config`

As we know the :class:`veles.znicz.standard_workflow.StandardWorkflow` was designed to simplify life of the inexperienced user.
In our code the :class:`veles.znicz.standard_workflow.StandardWorkflow` passes configuration to the embedded Units during creation.

.. note:: We also took a liberty to use root Config to define the topology of the
   desired neural network defined by :class:`veles.znicz.standard_workflow.StandardWorkflow`. Such flexible usage of the
   configuration should not confuse the User during creation of the custom Workflow.

Here is an example of configuration file with basic parameters of workflow and units, which you can
change in :class:`veles.znicz.standard_workflow.StandardWorkflow`. Also you can see Units parameters in :doc:`manualrst_veles_units_kwargs`

.. code-block:: python

  from veles.config import root


  root.common.precision_type = "float"
  root.common.precision_level = 1
  root.common.engine.backend = "cuda"

  root.my_workflow.update({
      "loss_function": "softmax",
      "loader_name": "lmdb",
      "loader": {"minibatch_size": 100,
                 "normalization_type": "external_mean",
                 "shuffle_limit": 1,
                 "train_paths": ["/home/Desktop/MyData/train"],
                 "validation_paths": ["/home/Desktop/MyData/validation"],
                 "crop": (227, 227),
                 "scale": (256, 256),
                 "color_space": "GRAY"},
      "layers": [{"type": "conv",
                  "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                         "padding": (2, 2, 2, 2), "sliding": (1, 1),
                         "weights_filling": "gaussian", "weights_stddev": 0.0001,
                         "bias_filling": "constant", "bias_stddev": 0},
                  "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                         "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                         "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  },
                 {"type": "max_pooling",
                  "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                 {"type": "activation_str"},

                 {"type": "norm", "alpha": 0.00005,
                  "beta": 0.75, "n": 3, "k": 1},

                 {"type": "softmax",
                  "->": {"output_sample_shape": 10,
                         "weights_filling": "gaussian", "weights_stddev": 0.01,
                         "bias_filling": "constant", "bias_stddev": 0},
                  "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                         "weights_decay": 1.0, "weights_decay_bias": 0,
                         "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}],
      "decision": {"fail_iterations": 10, "max_epochs": 1000},
      "snapshotter": {"prefix": "my_workflow", "interval": 1,
                      "time_interval": 0},
      "image_saver": {"out_dirs":
                      ["/tmp/test", "/tmp/validation", "/tmp/train"]},
      "weights_plotter": {"limit": 64}
      })

  root.my_workflow.loader.normalization_parameters = {
      "mean_source": "/home/Desktop/MyData/mean_image.png")}


Common parameters
-----------------

You can change common parameters at ``root.common`` (we use keyword `common` as convention).

.. code-block:: python

   root.common.precision_type = "float"
   root.common.precision_level = 1
   root.common.engine.backend = "cuda"

Most importants common parameters are:

1. `precision_type` parameter is "float" or "double". Default value is "double".
2. `precision_level` parameter specified accuracy of calculation. 0 value is for use simple summation. 1 value is for use Kahan summation (9% slower). 2 value is for use multipartials summation (90% slower). Default value is 0.
3. `engine.backend` parameter sets backend. It could be "ocl" (for OpenCL), "cuda" (for CUDA) or "auto" (first, try to run on CUDA backend, then on OpenCL, then without acceleration). Default value is "auto".

List of all common parameters: :doc:`manualrst_veles_common_parameters`

Loss function parameter
-----------------------

You can change loss function parameter at
``root.name_of_your_workflow.loss_function``.


One of `loss_function` and (`decision_name`, `evaluator_name`) is obligatory
for :class:`veles.znicz.standard_workflow.StandardWorkflow`.

.. code-block:: python

  root.my_workflow.update({"loss_function": "softmax"})

`loss_function` parameter defines the Loss function.
All loss functions:

1. "softmax" - Softmax Loss function. Multinomial logistic regression: used for predicting a single class of K mutually exclusive classes.
2. "mse" - Mean squared error Loss function. MSE of an estimator measures the average of the squares of the "errors".

Decision name parameter
-----------------------

You can change `decision_name` parameter at
``root.name_of_your_workflow.decision_name``.

One of `loss_function` and `decision_name` is obligatory
for :class:`veles.znicz.standard_workflow.StandardWorkflow`.

If `loss_function` and `decision_name` are defined both, `loss_function` parameter will be ignored.

if `loss_function` is defined and `decision_name` is not, `decision_name` will be created automaticly.

.. code-block:: python

  root.my_workflow.update({"decision_name": "decision_mse"})

Here is a list of all decision, which can be used.

1. "decision_mse" - :class:`veles.znicz.decision.DecisionMSE`. Correspond to "mse" `loss_function`.
2. "decision_gd" - :class:`veles.znicz.decision.DecisionGD`. Correspond to "softmax" `loss_function`.

Evaluator name parameter
------------------------

You can change evaluator name parameter at
``root.name_of_your_workflow.evaluator_name``.

One of `loss_function` and `evaluator_name` is obligatory
for :class:`veles.znicz.standard_workflow.StandardWorkflow`.

If `loss_function` and `evaluator_name` are defined both, `loss_function` parameter will be ignored.

if `loss_function` is defined and `evaluator_name` is not, `evaluator_name` will be created automaticly.

.. code-block:: python

  root.my_workflow.update({"evaluator_name": "evaluator_mse"})

Here is a list of all evaluators, which can be used.

1. "evaluator_mse" - :class:`veles.znicz.evaluator.EvaluatorMSE`. Correspond to "mse" `loss_function`.
2. "evaluator_softmax" - :class:`veles.znicz.evaluator.EvaluatorSoftmax`. Correspond to "softmax" `loss_function`.

Loader name parameter
---------------------

You can change loader name parameter at
``root.name_of_your_workflow.loader_name``.

`loader_name` is obligator parameter for :class:`veles.znicz.standard_workflow.StandardWorkflow`.

.. code-block:: python

  root.my_workflow.update({"loader_name": "lmdb"})

`loader_name` parameter is define name of loader, which will read the data.

Here is a list of all loaders, which can be used without redefining any functions.

1. "file_list_image" - :class:`veles.loader.file_image.FileListImageLoader`
2. "auto_label_file_image" - :class:`veles.loader.file_image.AutoLabelFileImageLoader`
3. "full_batch_pickles_image" - :class:`veles.loader.pickles.PicklesImageFullBatchLoader`
4. "full_batch_file_list_image" - :class:`veles.loader.fullbatch_image.FullBatchFileListImageLoader`
5. "full_batch_auto_label_file_image" - :class:`veles.loader.fullbatch_image.FullBatchAutoLabelFileImageLoader`
6. "full_batch_auto_label_file_image_mse" - :class:`veles.loader.fullbatch_image.FullBatchAutoLabelFileImageLoaderMSE`
7. "lmdb" - :class:`veles.znicz.loader.loader_lmdb.LMDBLoader`

You can see a list of all Loaders with parameters here: :doc:`manualrst_veles_units_kwargs`

Basic parameters of units
-------------------------

Here is a list of parameters, which can be defined for any Unit (e.g, Loader,
Decision)

You can change basic units parameters at ``root.name_of_your_workflow.name_of_unit``.

.. code-block:: python

  root.my_workflow.update({"loader": {"force_numpy": False}})

1. `force_numpy` forces the unit to use “numpy” backend, that is, disables any acceleration.
2. `generate_data_for_slave_threadsafe`  - value indicating whether generate_data_for_slave() method is invoked in a thread safe manner (under a mutex).
3. `name` - unit name, a string value which distinguishes it from the others. If it was not explicitly specified, the corresponding class name is returned. Name may be not unique, so if you need to map units, use :property`id` instead.
4. `logger` - the logging.Logger instance used for logging.
5. `view_group` - string key which defines this unit’s style (particularly, color) in workflow graphs. See :attr:`veles.workflow.Workflow.VIEW_GROUP_COLORS`
6. `apply_data_from_slave_threadsafe` - value indicating whether apply_data_from_slave() method is invoked in a thread safe manner (under a mutex).
7. `timings` - value indicating whether this unit should print run time statistics after each :method`run()` invocation. If it is not defined in the constructor, the default value is set. The default value is True if this unit’s class is in root.common.timings set and False otherwise.
8. `cache` - value indicating whether to save the compiled acceleration code on disk for faster following initializations.

Data parameters
---------------

Data parameters are defined for Loaders units (:class:`veles.loader.base.Loader`
descendants). You can see a list of all Loaders with parameters here:
:doc:`manualrst_veles_units_kwargs`
You can change data parameters at ``root.name_of_your_workflow.loader`` .

.. code-block:: python

  root.my_workflow.update({
      "loader": {"minibatch_size": 100,
                 "normalization_type": "external_mean",
                 "shuffle_limit": 1,
                 "train_paths": ["/home/Desktop/MyData/train"],
                 "validation_paths": ["/home/Desktop/MyData/validation"],
                 "crop": (227, 227),
                 "scale": (256, 256),
                 "color_space": "GRAY"}})

  root.my_workflow.loader.normalization_parameters = {
      "mean_source": "/home/Desktop/MyData/mean_image.png")}

Here are some parameters of different Loaders:

1. `prng` - sets random seed in shuffling the data. Default value is random_generator.get()
2. `normalization_type` - sets type of normalization loading data. Default value is "none". All normalization types (see at :mod:`veles.normalization`):

   1. "none" - :class:`veles.normalization.NoneNormalizer`

   2. "linear" - :class:`veles.normalization.LinearNormalizer`

   3. "mean_disp" - :class:`veles.normalization.MeanDispersionNormalizer`

   4. "exp" - :class:`veles.normalization.ExponentNormalizer`

   5. "pointwise" - :class:`veles.normalization.PointwiseNormalizer`

   6. "external_mean" - :class:`veles.normalization.ExternalMeanNormalizer`

   7. "internal_mean" - :class:`veles.normalization.InternalMeanNormalizer`

3. `normalization_parameters` - parameters for normalization. For example, "mean_source" must be defined for "external_mean" normalization.
4. `shuffle_limit` - sets limit of shuffling data. If shuffle_limit=-1: not shuffling. If shuffle_limit=1: shuffling data just one time (just like in Caffe). If shuffle_limit=numpy.iinfo(numpy.uint32).max: shuffle data every epoch. Default value is numpy.iinfo(numpy.uint32).max.
5. `minibatch_size` - sets size of one minibatch. Default value is 100
6. If `validation_ratio` is a number from 0 to 1, Loader automatically extracts a validation sample from train with that ratio. Default value is None.
7. `color_space`
8. `add_sobel`
9. `mirror`
10. `scale`
11. `scale_maintain_aspect_ratio`
12. `rotations`
13. `background_image`
14. `background_color`
15. `crop`
16. `smart_crop`
17. `crop_number`
18. `filename_types`
19. `ignored_files`
20. `included_files`
21. `path_to_test_text_file`
22. `path_to_val_text_file`
23. `path_to_train_text_file`
24. `test_paths`
25. `validation_paths`
26. `train_paths`
27. `label_regexp`
28. `target_paths`


Model structure and layer's parameters
--------------------------------------

There 2 ways to set topology: with `layers` parameter or with `mcdnnic_topology` parameter.

First way to set topology
.........................

Model's topology are defined for Forwards (:class:`veles.znicz.nn_units.ForwardBase`
descendants) and GDs units (:class:`veles.znicz.nn_units.GradientDescentBase`
descendants). You can change model's topology with parameters for **each** layer at
``root.name_of_your_workflow.layers``.

.. note:: You can set different parameters for each layer.

.. code-block:: python

  root.my_workflow.update({
      "layers": [{"type": "conv",
                  "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                         "padding": (2, 2, 2, 2), "sliding": (1, 1),
                         "weights_filling": "gaussian", "weights_stddev": 0.0001,
                         "bias_filling": "constant", "bias_stddev": 0},
                  "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                         "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                         "gradient_moment": 0.9, "gradient_moment_bias": 0.9},
                  },
                  {"type": "max_pooling",
                   "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                  {"type": "activation_str"},

                  {"type": "norm", "alpha": 0.00005,
                   "beta": 0.75, "n": 3, "k": 1},

                  {"type": "softmax",
                   "->": {"output_sample_shape": 10,
                          "weights_filling": "gaussian", "weights_stddev": 0.01,
                          "bias_filling": "constant", "bias_stddev": 0},
                   "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                          "weights_decay": 1.0, "weights_decay_bias": 0,
                          "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}]
                          })
.. note:: On last layer `output_sample_shape` (number of neurons) is not necessary. It will be created automatically by Loader`s number of labels (number of classes)

`layers` defines all structure of the Model with parameters for each layer.
`layers` is a list of dictionaries. Each dictionary sets each layer of the Model.
`type` parameter defines the type of layer. For example:

.. code-block:: python

  root.my_workflow.update({
      "layers": [{"type": "conv"},
                 {"type": "max_pooling"},
                 {"type": "activation_str"},
                 {"type": "norm"}
                 {"type": "softmax"}]
      })

The following code defines this Model's structure:

.. image:: _static/layers.png

Here is the list of all layer types:

1. "all2all_resizable" - (Forward: :class:`veles.znicz.resizable_all2all.ResizableAll2All`)
2. "all2all_tanh" - (Forward: :class:`veles.znicz.all2all.All2AllTanh`, Backward: :class:`veles.znicz.gd.GDTanh`)
3. "stochastic_abs_pool_depool" - (Forward: :class:`veles.znicz.pooling.StochasticAbsPoolingDepooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxPooling`)
4. "all2all_sigmoid" - (Forward: :class:`veles.znicz.all2all.All2AllSigmoid`, Backward: :class:`veles.znicz.gd.GDSigmoid`)
5. "activation_log" - (Forward: :class:`veles.znicz.activation.ForwardLog`, Backward: :class:`veles.znicz.activation.BackwardLog`)
6. "avg_pooling" - (Forward: :class:`veles.znicz.pooling.AvgPooling`, Backward: :class:`veles.znicz.gd_pooling.GDAvgPooling`)
7. "depooling" - (Forward: :class:`veles.znicz.depooling.Depooling`)
8. "channel_merger" - (Forward: :class:`veles.znicz.channel_splitting.ChannelMerger`)
9. "deconv" - (Forward: :class:`veles.znicz.deconv.Deconv`, Backward: :class:`veles.znicz.gd_deconv.GDDeconv`)
10. "activation_tanhlog" - (Forward: :class:`veles.znicz.activation.ForwardTanhLog`, Backward: :class:`veles.znicz.activation.BackwardTanhLog`)
11. "all2all_str" - (Forward: :class:`veles.znicz.all2all.All2AllStrictRELU`, Backward: :class:`veles.znicz.gd.GDStrictRELU`)
12. "activation_relu" - (Forward: :class:`veles.znicz.activation.ForwardRELU`, Backward: :class:`veles.znicz.activation.BackwardRELU`)
13. "maxabs_pooling" - (Forward: :class:`veles.znicz.pooling.MaxAbsPooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxAbsPooling`)
14. "rprop_all2all" - (Backward: :class:`veles.znicz.rprop_all2all.RPropAll2All`)
15. "stochastic_pooling" - (Forward: :class:`veles.znicz.pooling.StochasticPooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxPooling`)
16. "conv_str" - (Forward: :class:`veles.znicz.conv.ConvStrictRELU`, Backward: :class:`veles.znicz.gd_conv.GDStrictRELUConv`)
17. "channel_splitter" - (Forward: :class:`veles.znicz.channel_splitting.ChannelSplitter`)
18. "activation_str" - (Forward: :class:`veles.znicz.activation.ForwardStrictRELU`, Backward: :class:`veles.znicz.activation.BackwardStrictRELU`)
19. "activation_tanh" - (Forward: :class:`veles.znicz.activation.ForwardTanh`, Backward: :class:`veles.znicz.activation.BackwardTanh`)
20. "activation_sincos" - (Forward: :class:`veles.znicz.activation.ForwardSinCos`, Backward: :class:`veles.znicz.activation.BackwardSinCos`)
21. "dropout" - (Forward: :class:`veles.znicz.dropout.DropoutForward`,Backward:  :class:`veles.znicz.dropout.DropoutBackward`)
22. "cutter" - (Forward: :class:`veles.znicz.cutter.Cutter`, Backward: :class:`veles.znicz.cutter.GDCutter`)
23. "conv_sigmoid" - (Forward: :class:`veles.znicz.conv.ConvSigmoid`, Backward: :class:`veles.znicz.gd_conv.GDSigmoidConv`)
24. "max_pooling" - (Forward: :class:`veles.znicz.pooling.MaxPooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxPooling`)
25. "activation_mul" - (Forward: :class:`veles.znicz.activation.ForwardMul`, Backward: :class:`veles.znicz.activation.BackwardMul`)
26. "conv" - (Forward: :class:`veles.znicz.conv.Conv`, Backward: :class:`veles.znicz.gd_conv.GradientDescentConv`)
27. "softmax" - (Forward: :class:`veles.znicz.all2all.All2AllSoftmax`, Backward: :class:`veles.znicz.gd.GDSoftmax`)
28. "all2all" - (Forward: :class:`veles.znicz.all2all.All2All`, Backward: :class:`veles.znicz.gd.GradientDescent`)
29. "norm" - (Forward: :class:`veles.znicz.normalization.LRNormalizerForward`, Backward: :class:`veles.znicz.normalization.LRNormalizerBackward`)
30. "all2all_relu" - (Forward: :class:`veles.znicz.all2all.All2AllRELU`, Backward: :class:`veles.znicz.gd.GDRELU`)
31. "zero_filter" - (Forward: :class:`veles.znicz.weights_zerofilling.ZeroFiller`)
32. "stochastic_abs_pooling" - (Forward: :class:`veles.znicz.pooling.StochasticAbsPooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxAbsPooling`)
33. "conv_tanh" - (Forward: :class:`veles.znicz.conv.ConvTanh`, Backward: :class:`veles.znicz.gd_conv.GDTanhConv`)
34. "stochastic_pool_depool" - (Forward: :class:`veles.znicz.pooling.StochasticPoolingDepooling`, Backward: :class:`veles.znicz.gd_pooling.GDMaxPooling`)
35. "activation_sigmoid" - (Forward: :class:`veles.znicz.activation.ForwardSigmoid`, Backward: :class:`veles.znicz.activation.BackwardSigmoid`)
36. "conv_relu" - (Forward: :class:`veles.znicz.conv.ConvRELU`, Backward: :class:`veles.znicz.gd_conv.GDRELUConv`)

Symbols `->` setting parameters for forward propagation.

.. code-block:: python

  root.my_workflow.update({
      "layers": [{"type": "conv",
                  "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                         "padding": (2, 2, 2, 2), "sliding": (1, 1),
                         "weights_filling": "gaussian", "weights_stddev": 0.0001,
                         "bias_filling": "constant", "bias_stddev": 0}}]
                         })

Here are some of forward propagation parameters:

1. `kx`
2. `weights_stddev`
3. `include_bias`
4. `n_kernels`
5. `bias_stddev`
6. `bias_filling`
7. `unpack_size`
8. `ky`
9. `sliding`
10. `rand`
11. `padding`
12. `weights_filling`
13. `weights_transposed`
14. `output_dtype`
15. `output_sample_shape`
16. `output_samples_number`
17. `unsafe_padding`
18. `grouping`
19. `beta`
20. `k`
21. `n`
22. `alpha`

Symbols `<-` setting parameters for backward propagation.

.. code-block:: python

  root.my_workflow.update({
      "layers": [{"type": "conv",
                  "<-": {"learning_rate": 0.001, "learning_rate_bias": 0.002,
                         "weights_decay": 0.0005, "weights_decay_bias": 0.0005,
                         "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}]
                         })

Here are some of backward propagation parameters:

1. `include_bias`
2. `weights_decay_bias`
3. `last_minibatch`
4. `factor_ortho`
5. `fast_learning_rate`
6. `gradient_moment`
7. `accumulate_gradient`
8. `weights_transposed`
9. `variant_gradient`
10. `need_err_input`
11. `adadelta_momentum`
12. `variant_moment_gradient`
13. `adagrad_epsilon`
14. `adadelta_epsilon`
15. `l1_vs_l2`
16. `learning_rate`
17. `adadelta_adom`
18. `gradient_moment_bias`
19. `weights_decay`
20. `solvers`
21. `l1_vs_l2_bias`
22. `learning_rate_bias`

Second way to set topology
..........................

`mcdnnic_topology` is a way to set topology like
in artical http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.
You can change model's topology at ``root.name_of_your_workflow.mcdnnic_topology``.
And set parameters for all layers at ``root.name_of_your_workflow.mcdnnic_parameters`` with `mcdnnic_parameters` parameter.

.. note:: Parameters of layers with `mcdnnic_parameters` are the same for each layer.

.. code-block:: python

  root.my_workflow.update({
      "mcdnnic_topology": "12x256x256-32C4-MP2-64C4-MP3-32N-4N"})

  root.lines.mcdnnic_parameters = {
      "->": {"weights_filling": "uniform", "weights_stddev": 0.05,
             "bias_filling": "constant", "bias_stddev": 0},
      "<-": {"learning_rate": 0.03, "learning_rate_bias": 0.002,
             "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}

Timing parameters
-----------------

Timing parameters are defined for Decision (:class:`veles.znicz.decision.DecisionBase`
descendant) unit. You can change timing parameters at
``root.name_of_your_workflow.decision``.

.. code-block:: python

  root.my_workflow.update({
      "decision": {"fail_iterations": 10, "max_epochs": 1000}})

1. `max_epochs`
2. `fail_iterations`

Snapshotting parameters
-----------------------

Snapshotting parameters are defined for Snapshotter (
:class:`veles.snapshotter.SnapshotterBase` descendant) unit.
You can change snapshotting parameters at ``root.name_of_your_workflow.snapshotter``.

.. code-block:: python

  root.my_workflow.update({
      "snapshotter": {"prefix": "my_workflow", "interval": 1,
                      "time_interval": 0}})

1. `time_interval`
2. `compress_level`
3. `directory`
4. `prefix`
5. `interval`

Evaluation parameters
---------------------

Evaluation parameters are defined for Evaluator (
:class:`veles.znicz.evaluator.EvaluatorBase` descendant) unit.
You can change evaluation parameters at ``root.name_of_your_workflow.evaluator``.

.. code-block:: python

  root.my_workflow.update({
      "evaluator": {"": }
      })

1. `root`
2. `mean`
3. `compute_confusion_matrix`

LearningRateAdjuster parameters
-------------------------------

LearningRateAdjuster (:class:`veles.znicz.lr_adjust.LearningRateAdjust`) parameters are
defined at ``root.name_of_your_workflow.lr_adjuster``.

.. code-block:: python

  root.my_workflow.lr_adjuster.lr_parameters = {
      "lrs_with_lengths": [(1, 60000), (0.1, 5000), (0.01, 100000000)]}
  root.my_workflow.lr_adjuster.bias_lr_parameters = {
      "lrs_with_lengths": [(1, 60000), (0.1, 5000), (0.01, 100000000)]}

  root.my_workflow.update({
      "lr_adjuster": {"lr_policy_name": "arbitrary_step",
                      "bias_lr_policy_name": "arbitrary_step"}})

1. `lr_policy_name`: "exp", "fixed", "step_exp", "inv", "arbitrary_step"
2. `bias_lr_policy_name`: "exp", "fixed", "step_exp", "inv", "arbitrary_step"
3. `lr_parameters`
4. `bias_lr_parameters`

Here is a list of LRAdjusterPolicy classes:

1. "exp" - :class:`veles.znicz.lr_adjust.ExpPolicy`
2. "fixed" - :class:`veles.znicz.lr_adjust.FixedAjustPolicy`
3. "step_exp" - :class:`veles.znicz.lr_adjust.StepExpPolicy`
4. "inv" - :class:`veles.znicz.lr_adjust.InvAdjustPolicy`
5. "arbitrary_step" - :class:`veles.znicz.lr_adjust.ArbitraryStepPolicy`

Other units parameters
----------------------

Here is example of parameters for :class:`veles.znicz.image_saver.ImageSaver`
and :class:`veles.znicz.nn_plotting_units.Weights2D`:

.. code-block:: python

  root.my_workflow.update({
      "image_saver": {"out_dirs":
                      ["/tmp/test", "/tmp/validation", "/tmp/train"]},
      "weights_plotter": {"limit": 64}
      })

1. `out_dirs`
2. `limit`

Here is a list of all other units with parameters: :doc:`manualrst_veles_units_kwargs`