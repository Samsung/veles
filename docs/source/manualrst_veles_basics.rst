======
Basics
======

VELES uses modular paradigm for quick and easy development of Machine Learning
algorithms and models.

.. image:: _static/units_to_workflow.png

User can construct any dataflow algorithm including Neural Network models using
the predefined elementary building blocks - Units.


.. _two_ways_of_running_veles:

What is a Model?
::::::::::::::::

The Model or Workflow - a collection of Units connected to each other
implementing required algorithms. The class :class:`veles.workflow.Workflow`
is the container for the Model.

The Workflow can be very complex and can execute algorithms iteratively on the dataset.
In order to set start and end points in the algorithms the Workflow has a start
point and an end point (:class:`StartPoint <veles.workflow.StartPoint>`
and :class:`EndPoint <veles.workflow.EndPoint>`), so that when
:meth:`run() <veles.workflow.Workflow.run>` is called, the start point begins
the party and the end point's run ends the party, triggering
:meth:`on_workflow_finished() <veles.workflow.Workflow.on_workflow_finished>`.

Two ways of running Veles
:::::::::::::::::::::::::

You can start Veles by two methods:

    * ``python3 -m veles ...``
    * ``veles ...`` (**only** in case of :doc:`manualrst_veles_ubuntu_user_setup`)
    
Both methods execute :mod:`veles.__main__`, and they are absolutely equivalent.

.. include:: manualrst_veles_modes.rst

.. include:: manualrst_veles_units.rst