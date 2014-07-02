"""
Created on Jul 2, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from veles.units import UnitRegistry


class CommandLineArgumentsRegistry(type):
    """
    Metaclass to accumulate command line options from scattered classes for
    velescli's upmost argparse.
    """
    classes = []

    def __init__(cls, name, bases, clsdict):
        # if the class does not have it's own init_parser(), no-op
        init_parser = clsdict.get('init_parser', None)
        if init_parser is None:
            return
        # early check for the method existence
        if not isinstance(init_parser, staticmethod):
            raise TypeError("init_parser must be a static method since the "
                            "class has CommandLineArgumentsRegistry metaclass")
        CommandLineArgumentsRegistry.classes.append(cls)
        super(CommandLineArgumentsRegistry, cls).__init__(name, bases, clsdict)


class UnitCommandLineArgumentsRegistry(UnitRegistry,
                                       CommandLineArgumentsRegistry):
    """
    Enables the usage of CommandLineArgumentsRegistry with classes derived from
    Unit.
    """
    pass
