=============
PyCharm setup
=============

VELES can be developed in either Eclipse PyDev IDE or PyCharm IDE.
If you choose PyCharm, the first step is downloading and running it. Grab Pycharm
from `the official site <https://www.jetbrains.com/pycharm/download/>`_.

To run PyCharm you will need a JRE. Standard OpenJDK is enough. In Ubuntu,
it is installable as ``openjdk-7-jre`` package.

General settings
::::::::::::::::

^^^^^^^^^^^^^^^^^^^^^^^^
Linux font rendering fix
^^^^^^^^^^^^^^^^^^^^^^^^

As of 2014, PyCharm's Swing default font rendering on Linux makes blood trickle from your eyes.
To make it better (but not the best), append the following lines to ``bin/pycharm64.vmoptions``::

    -Dswing.aatext=true
    -Dawt.useSystemAAFontSettings=on
    -Dsun.java2d.xrender=true

and remove::

    -Dawt.useSystemAAFontSettings=lcd

^^^^^^^^^^
CodeGlance
^^^^^^^^^^

You should install CodeGlance plugin (**File -> Settings... -> Plugins -> Browse Repository**).

^^^^^^^^^^^^^^^
Import settings
^^^^^^^^^^^^^^^

There is an "all-in-one" configuration which can be imported (tested on 3.4.1):
:download:`download <_static/pycharm-3.4.1-settings.jar>`.