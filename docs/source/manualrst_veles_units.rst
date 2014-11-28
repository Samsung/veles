:orphan:

Units
:::::

*Unit* is a core concept of VELES. Technically, it is class inherited from :class:`Unit <veles.units.Unit>`
and satisfying :class:`IUnit <veles.units.IUnit>` interface.

A unit carries some algorithm which is solid and indivisible - it is a *building block*. For example, it is
a single layer of a neural network or a rhythm extraction from music data. A unit
can be *constructed*, *initialized* and *run*. When a unit is constructed, it 
creates it's *output* data fields. During the initialization, *input* data fields
are validated and some preparation is done for running. Unit's run is a reentrant
application of the underlying algorithm, which processes inputs and updates outputs.
Units connect with each other in two ways: control flow links and data links.

.. image:: _static/unit.png

The blue unit will not run unless all the three orange predecessors finish to run.
This is a *control flow*. The blue unit takes inputs from orange units' outputs.
This makes units' running *data driven*.

Units may unite into a :class:`Workflow <veles.workflow.Workflow>`. Actually, each unit
has a :attr:`workflow` property which points to the parent object. Workflows are
units, too, so they can also be linked and run, but in a slightly different way.
See :doc:`manualrst_veles_workflow`.

Distributed calculation is performed using the additional group of methods required
by :class:`IDistributable <veles.distributable.IDistributable>` (see :doc:`manualrst_veles_distributed_units`).

------------------
How to link Units?
------------------


Control flow
~~~~~~~~~~~~
To make one unit run after another, use :py:meth:`veles.units.Unit.link_from` ::

    second_unit.link_from(first_unit)

To see, after whom is given unit run, use  :py:meth:`veles.units.Unit.links_from` 

To remove control flow link (not data link), use :py:meth:`veles.units.Unit.unlink_from` and :py:meth:`veles.units.Unit.unlink_all`


Data flow
~~~~~~~~~
For example, if you want to pass convolutional layer output to pooling layer input ::

    # conv_layer.output --> pooling_layer.input
    pooling_layer.link_attrs(conv_layer, ("input", "output")) 

The passing data will be **shared** between those units.


-------------
Service units
-------------

TODO: move Znicz units into znicz docs.

* `Loaders` load raw images, pre-process them and make the initial data vectors.
* `Repeater` is a dummy unit that should be linked from `start_point` and from the last unit of the Workflow.
* `Decision` decides whether to stop training or continue.
* `Snapshotter` makes `pickle` snapshots from the `Workflow` each epoch.
* `Plotters` are used to draw plots: weight matrices, error for epochs, etc.
