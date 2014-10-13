=============
Eclipse setup
=============

.. sidebar:: Adding Eclipse icon in Ubuntu/Unity

   .. rubric:: Desktop file
   
   Create ``/usr/share/applications/eclipse.desktop`` with the following content::

          [Desktop Entry]
          Comment=Eclipse CDT Luna
          Terminal=false
          Name=Eclipse
          Exec=env UBUNTU_MENUPROXY= /opt/eclipse/eclipse
          Type=Application
          Categories=IDE;Development
          X-Ayatana-Desktop-Shortcuts=NewWindow
          Icon=/opt/eclipse/icon.xpm
            
          [NewWindow Shortcut Group]
          Name=New Window
          Exec=env UBUNTU_MENUPROXY= /opt/eclipse/eclipse
          TargetEnvironment=Unity
          
   You should change the values according to where you installed Eclipse,
   latest Eclipse version name and whether the nasty bug with global menu is fixed.
    
   .. rubric:: Icon in sidebar
   
   Open Dash and type "eclipse". Drag and drop the appeared icon to the sidebar.
    

VELES is being (but not strictly has to be) developed in Eclipse PyDev IDE.
So to start developing VELES, the first step is downloading and running Eclipse.

VELES team mebers should already have it in /opt/eclipse, others can go to
http://eclipse.org/downloads. Using Eclipse CDT (C/C++ IDE) as the base is recommended.
**In Ubuntu, usage of system packages is discouraged**, because they are often outdated.

To run Eclipse you will need a JRE. Standard OpenJDK is enough. In Ubuntu,
it is installable as ``openjdk-7-jre`` package.

General settings
::::::::::::::::

    * Set Visual Studio key scheme. Go to **Main Menu -> Window -> Preferences -> General -> Keys**
      and select the "Microsoft Visual Studio" scheme.
    
    * Add 79 chars border. Go to **Main Menu -> Window -> Preferences -> General -> Editors -> Text Editors**,
      set "Print margin column" to 79 (PEP8 requirement) and check "Show print margin".
      Unfortunately, this setting is not specific to any perpective or editor, so
      the line is drawn in, say, C++ or XML editors as well.


PyDev
:::::

Go to **Main Menu -> Help -> Eclipse Marketplace**. In "Find:" line, type in
"python" and install **PyDev - Python IDE for Eclipse**.

^^^^^^^^^^^
Interpreter
^^^^^^^^^^^

Go to **Main Menu -> Window -> Preferences -> PyDev -> Interpreters -> Python Interpreter**,
remove any existing interpreters, click "New...". Set "python" as Name and ``/usr/bin/python3`` as Executable.
Select "Forced Builtins" tab and click "New...". Insert ``twisted,numpy,matplotlib,zmq``.

^^^^^^^^^^
Code style
^^^^^^^^^^

Active pep8. Go to **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Code Analysis**
and select "pep8.py" tab. Check "Error" instead of "Don't run".

Activate pylint. First, install it ``sudo apt-get install pylint``, and 
then go to **Main Menu -> Window -> Preferences -> PyDev -> PyLint**, check
"Use PyLint?" and insert ``/usr/bin/pylint`` to "Location of the pylint executable".
Insert the following exception to "Arguments to pass to the pylint command"::

    --disable=E0611,E0001,W0632,C0103,C0111,C0325,W0201,R0911,R0913,R0914,R0915,
    W1401,R0901,W0108,R0902,W0221,R0201,W0104,W0602,W0105,R0912,E1002,W0212,W0101,
    W0613,W0222,W0603,W0211,W0612,E1123,W0703,E0202,W0142,R0903,W0223,W0622,W0511,
    W0122,W0702,W0107,W1201,R0904,F0401,E1103,E1101,W0141,W0232,E0211,E0213

Go to **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Code Style -> Code Formatter**
You should check the following **only**:

    * Auto-format only files in the workspace
    * Use space after commas
    * Use space before and after operators (+,-,/,*,//,**,etc.)
    * Right trim lines
    * Right trim multi-line string literals
    * Add new line at the end of file
    * Spaces before a comment [2 spaces]
    * Spaces in comment start [At least 1 space]
    
Optionally, use autopep8.

Go to **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Save Actions**.
Check "Auto-format editor contents before saving".

^^^^^^^^^
Templates
^^^^^^^^^

Go to **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Templates** and
for each "Module: \*" (e.g., "Module: Class"), change the header to::
    
    """
    Created on ${date}
      
    Copyright (c) 2014 Samsung Electronics Co., Ltd.
    """

^^^^^
Other
^^^^^

Go to **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Overview Ruler Minimap**
and check "Show minimap".

ReST
::::

Finally, in order to easily write documentation, install Eclipse ReST Editor plugin.
Go to **Main Menu -> Help -> Eclipse Marketplace**. In "Find:" line, type in "rest editor".

After installation, \*.rst files will be opened in special editor with (kind of) autocompletion.
