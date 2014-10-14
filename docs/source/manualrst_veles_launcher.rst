Veles Launcher
::::::::::::::

Basically, :class:`Launcher <veles.launcher.Launcher>` mimics some :class:`Workflow <veles.workflow.Workflow>`
properties and methods, notably :meth:`run()`, :meth:`stop()` and :meth:`add_ref()`. This class is responsible for
running :class:`Server <veles.server.Server>` (in master mode), :class:`Client <veles.client.Client>` (in slave mode)
and the workflow. Besides, it starts :class:`GraphicsServer <veles.graphics_server.GraphicsServer>`, unless ``-p ''``
(empty Matplotlib backend) was specified in the command line and :class:`GraphicsClient <veles.graphics_client.GraphicsClient>`,
unless ``--no-graphics-client`` was specified. See :doc:`manualrst_veles_graphics` for more information about
how plotting subsystem works.

Your workflow instance's workflow property always points to :class:`Launcher` object,
even in :meth:`__init__()` (after :meth:`super()` call).

Launcher defines three modes of execution: standalone (default), master and slave.
Each mode can be tested with corresponding properties: :meth:`is_standalone`,
:meth:`is_master` and :meth:`is_slave`, which are mutually exclusive.