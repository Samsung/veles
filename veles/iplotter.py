"""
Created on Nov 5, 2014

Interface which states that the object supports plotting in GraphicsClient
environment.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from zope.interface import Interface, Attribute


class IPlotter(Interface):
    """Interface which states that the object supports plotting in
    GraphicsClient environment.
    """

    matplotlib = Attribute("""matplotlib module reference""")
    cm = Attribute("""matplotlib.cm (colormap) module reference""")
    lines = Attribute("""matplotlib.lines module reference""")
    patches = Attribute("""matplotlib.patches module reference""")
    pp = Attribute("""matplotlib.pyplot module reference""")

    def redraw():
        """Updates the plot using the changed object's state.
        Should be implemented by the class providing this interface.
        """
