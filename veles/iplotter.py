# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Nov 5, 2014

Interface which states that the object supports plotting in GraphicsClient
environment.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
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
