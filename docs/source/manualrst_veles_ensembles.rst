==================
Training ensembles
==================

Veles automates the process of training
`ensembles <https://en.wikipedia.org/wiki/Ensemble_learning>`_. It consists of
3 separate steps:

1. Train the models which are be included into the ensemble.
2. Evaluate those models on a separate part of the dataset (this ensures that
   the ensemble does not adapt to the validation set).
3. Train the top-level classifier on a separate part of the dataset which uses
   the output from step 2 as features.

(1) How to train the models
:::::::::::::::::::::::::::

The following command::

   veles -s --ensemble-train 20:0.9 --result-file ensemble.json <workflow> <config>

will result in 20 separate workflows, each being trained on 0.9 part of the
training dataset. The information about the ensemble, such as best snapshots,
evaluated metrics, etc., is saved to ensemble.json. As usual, multiple models
can be trained in parallel via master-slave.

Internally, Veles launches an instance of
:class:`veles.ensemble.model_workflow.EnsembleModelWorkflow` instead of the user's model.
It is linked in a ring and :class:`veles.ensemble.model_workflow.EnsembleModelManager`
unit trains one model in :func:`run()` at a time. This workflow contains the histogram
plotter which depicts the distribution of "EvaluationResult" metric value.

By default, plotting and publishing is disabled in workflows included into the
ensemble. If plotters are desired to work, set
``root.common.ensemble.disable.plotting`` to False::

   veles ... root.common.ensemble.disable.plotting=False

(2) How to evaluate the models
::::::::::::::::::::::::::::::

The following command::

   veles -s --ensemble-test ensemble.json --result-file ensemble_ev.json <workflow> <config>

writes the results of the evaluation of models trained on step 1 on test dataset
to ensemble_ev.json. Each model is restored from the snapshot, runs in "test" mode
(see :doc:`manualrst_veles_modes`) and appends the output to ensemble_ev.json.
Parallel evaluation via master-slave is supported. User's loader must support
"test" mode and supply the labelled (or targeted) data to TEST set. Hovewer,
labels or targets are not used in this step, they are needed only by step 3.

(3) How to train the top-level classifier
:::::::::::::::::::::::::::::::::::::::::

This step is not fully automated, unfortunately, because one can try different
classifers and the only way to allow this is to write a workflow. Nevertheless,
:class:`veles.loader.ensemble.EnsembleLoader` and :class:`veles.loader.ensemble.EnsembleLoaderMSE`
fullbatch loaders exist which load the results of step 2, that is, the information about
trained and evaluated models in the ensemble. User's loader should inherit from
**one** of them and implement :class:`veles.loader.ensemble.IEnsembleLoader` or
:class:`veles.loader.ensemble.IEnsembleLoaderMSE` interface, correspondingly.
Specifically, :func:`load_winners()` must return the list of the labels for the
test dataset used in step 2, either in index or raw forms, and
:func:`load_targets()` populates ``original_targets``.

Since the described steps are independent, one can generate the intermediate
files by hand. They are just plain text JSON files. Thus, it is possible to
combine different neural network topologies by merging ``--result-file``-s from
step 1, for example.
