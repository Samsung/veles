"""
Created on Apr 4, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from veles.units import Unit


class Calculator(Unit):
    """
    Calculates an arbitrary expression in calculate(), loading values from
    bound objects and setting them back afterwards.

    If any attribute of this class is a tuple (object, "name"), it is
    interpreted as a reference to object.name, so it is temporarily set to
    the dereferenced value before calculate() call and then  restored after
    calculate() is finished, updating the referenced value.
    """

    def __init__(self, workflow, **kwargs):
        super(Calculator, self).__init__(workflow, **kwargs)

    def run(self):
        """
        Invokes overriden calculate(), loading and saving referenced values.
        """
        args = {}
        for key, value in self.__dict__.items():
            if key in Unit.__dict__:
                continue
            if isinstance(value, tuple):
                args[key] = value
                setattr(self, key, getattr(*value))
        self.calculate()
        for key, value in args.items():
            setattr(value[0], value[1], getattr(self, key))
            setattr(self, key, value)

    def calculate(self, *args):
        """
        Inherited classes should override this.
        """
        pass
