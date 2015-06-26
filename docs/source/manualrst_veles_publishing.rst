==================
Publishing reports
==================

There are units for publishing reports about workflow runs. In other words,
one can insert a properly configured :class:`veles.publishing.publisher.Publisher`
unit before :class:`veles.plumbing.EndPoint` to write HTML or PDF or online wiki
page with very detailed information what input data and the configuration
were specified, how the workflow ran, what metric values were achieved and what
plots were drawn. Currently, only Confluence publishing backend is supported,
but more backends will eventually emerge later.

If you are using :class:`veles.znicz.standard_workflow.StandardWorkflow`, using
Publisher is as simple as calling::

   self.link_publisher(<parent>)

inside :func:`veles.znicz.standard_workflow.StandardWorkflow.create_workflow`.
It will require some basic configuration which is backend-specific, for example,
this is for Confluence::

   "publisher": {
        "backends": {
            "confluence": {
                "server": "http://localhost:8000",
                "username": "user", "password": "password",
                "space": "VEL", "parent": "Veles"
            }
        }
   }

A typical report consists of several sections.

#. General information about the workflow, containing the image by the path which is taken from the manifest file, if it exists. Manifest file is JSON metadata required by VelesForge, so refer to :doc:`manualrst_veles_forge`.
#. Achieved results. The values in that table are taken from units which implement :class:`veles.result_provider.IResultProvider` interface. For example, achieved best accuracy, RMSE, total number of epochs, etc.
#. Source data basic analytics: test, validation and train sets ratio and sizes. Label distribution in case of classification task. Normalization type and parameters used.
#. Run statistics: elapsed time, unit run time profile. Talking about unit run times, one should remember that GPU accelerated units use asunchronous GPU pipeline, returning the control at once, so to see the real profile, pass ``--sync-run`` command line argument.
#. The configuration used.
#. Random seeds.
#. Workflow scheme as if it was written by ``--workflow-graph``.
#. Plots.

Publisher can be extended with more report data by overriding
:func:`veles.publishing.publisher.Publisher.gather_info` method and adding the corresponding
support to the used backends.
