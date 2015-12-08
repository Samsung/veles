========
Advanced
========

Command line startup process
::::::::::::::::::::::::::::

Here is the description of what happens after you execute veles:

    #. A new instance of :class:`__main__.Main` class is created and
       it's :func:`run() <__main__.Main.run>` method is called.
       If :meth:`run()` returns, the resulting value is used as the return code for
       :func:`sys.exit()`.
    #. "Special" command line arguments are checked for presence. They include
       ``--help``, ``--html-help`` and ``--frontend``.
    #. :mod:`argparse`-based parser is initialized and parses :attr:`sys.argv`.
       See :doc:`manualrst_veles_cmdline_system`.
    #. Logging level is set (see :mod:`-v / --verbosity option <scripts.velescli>`). :data:`DEBUG`
       log level is set for classes enumerated in ``--debug``.
    #. Random generator is seeded with the value of ``-r / --random-seed`` option, if it was specified. Otherwise,
       16 numbers from ``/dev/urandom`` are used.
    #. Pickle debugging is activated, if ``--debug-pickle`` was specified (see :doc:`manualrst_veles_pickles`).
    #. Peak memory usage printer is registered on program exit.
    #. The specified workflow file is imported.
    #. The specified configuration file is imported. Command line overrides the
       current configuration tree (see :class:`veles.config.Config`). If ``--dump-config``
       option was specified, the resulting configuration is printed to standard output.
    #. Special configuration parameters are parsed from command line and configuration file is override.
    #. If ``--dry-run`` was set to "load", execution finishes just after instantiation.
    #. Otherwise, if ``-b / --background`` was passed, the process turns into a
       `daemon <https://en.wikipedia.org/wiki/Daemon_(computing)>`_ process,
       which is independent of it's parent. Many operations are performed in that case,
       most notably fork()-ing.
    #. If ``--optimize`` was passed, the parameters optimization procedure begins
       via the genetic algorithm, see :doc:`manualrst_veles_genetic_optimization`.
    #. If --ensemble-train was passed, training of ensemble models is started.
    #. If --ensemble-test was passed, evaluation of ensemble models is started.
    #. Otherwise, :func:`run()` is called from the specified workflow module.
       Two :class:`Main` methods are executed :meth:`_load <scripts.Main._load>` and
       :meth:`_main <scripts.Main._main>`
    #. :meth:`_load` constructs an instance of :class:`Launcher <veles.launcher.Launcher>`
       creates the workflow (corresponding class is specified as the first function argument)
       or restores it from the snapshot file (``--snapshot``, see :doc:`manualrst_veles_snapshotting`).
       In the former case the launcher is passed as workflow's parent workflow,
       in the latter case the "workflow" property is set to the launcher.
       If ``--workflow-graph`` was passed, writes the workflow scheme using
       `Graphviz <https://en.wikipedia.org/wiki/Graphviz>`_.
    #. :meth:`_main` returns at once if ``--dry-run`` was set to "init", otherwise
       it creates an OpenCL/CUDA device handle, unless ``--force-numpy`` was passed.
       Then it calls :meth:`initialize() <veles.workflow.Workflow.initialize>` workflow's method,
       passing in the created OpenCL device. If ``--dry-run`` was set to "exec", returns.
       Finally, it calls :meth:`launcher's run() <veles.launcher.Launcher.run>`.

.. include:: manualrst_veles_launcher.rst
.. include:: manualrst_veles_cmdline_system.rst
.. include:: manualrst_veles_pickles.rst
.. include:: manualrst_veles_vector.rst
.. include:: manualrst_veles_using_configs.rst