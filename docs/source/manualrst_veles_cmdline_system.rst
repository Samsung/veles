:orphan:

Command line arguments organization
:::::::::::::::::::::::::::::::::::

Every class which has :class:`CommandLineArgumentsRegistry <veles.cmdline.CommandLineArgumentsRegistry>` as it's
metaclass (:class:`UnitCommandLineArgumentsRegistry <veles.units.UnitCommandLineArgumentsRegistry>` for classes
derived from :class:`Unit <veles.units.Unit>`) can define it's own command line
arguments which are included into the global list. Such class must have

.. method:: init_parser(**kwargs)
   :noindex:

defined, which first obtain the instance of :class:`argparse.ArgumentParser`::

    parser = kwargs.get("parser", argparse.ArgumentParser())
    
and then append arguments with :func:`parser.add_argument()` and return it.
When an instance of that class wants to find out the argument values,
it calls :func:`init_parser()` without any parameters and then
invoke :meth:`parse_known_args()`.

Internally, :class:`CommandLineArgumentsRegistry <veles.cmdline.CommandLineArgumentsRegistry>`
metaclass keeps the list of classes with additional command line arguments and
during `the startup <Command line startup process>`_ each class from that
list sequentially alters the same parser.