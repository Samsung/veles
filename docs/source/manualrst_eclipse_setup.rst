Eclipse setup
=============

* "Download page":http://www.eclipse.org/cdt/downloads.php
* "Download Kepler":http://www.eclipse.org/downloads/download.php?file=/technology/epp/downloads/release/kepler/R/eclipse-cpp-kepler-R-linux-gtk-x86_64.tar.gz

*   Install additional plugins for Eclipse. Go to **Main Menu -> Help -> Install New Software...** Select "Kepler - http://download.eclipse.org/releases/kepler" in "Work with:". Select **Linux Tools/OProfile Integration** and
    **Linux Tools/Valgrind Tools Integration**. Click Finish. Install **Autotools support for CDT** the same way. Add **"http://sourceforge.net/projects/shelled/files/shelled/update/"** as "ShellEd" to "Work with:" list and
    install ShellEd. Go to **Main Menu -> Help -> Eclipse Marketplace**. In "Find:" line, type in egit. Install **EGit - Git Team Provider**. Then type in "python" and install **PyDev - Python IDE for Eclipse**. Install **Rinzo XML Editor** as well.

* Set your ${user} in Eclipse. Edit /path/to/eclipse/eclipse.ini, insert line
::

    -Duser.name=Ivanov Ivan <i.ivanov@samsung.com>

after "-Dhelp.lucene.tokenizer=standard", so that the whole part is like
::

    -vmargs
    -Dosgi.requiredJavaVersion=1.5
    -Dhelp.lucene.tokenizer=standard
    -Duser.name=Ivanov Ivan <i.ivanov@samsung.com>
    -XX:MaxPermSize=256m
    -Xms40m
    -Xmx384m

* Set the default file header. Go to **Main Menu -> Window -> Preferences -> C/C++ -> Code Style -> Code Templates -> Comments -> Files**, click "Edit...". Insert the following:
::

    /*! @file ${file_name}
     *  @brief New file description.
     *  @author ${user}
     *  @version 1.0
     *
     *  @section Notes
     *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
     *
     *  @section Copyright
     *  Copyright 2013 Samsung R&D Institute Russia
     */

* Set the default header template. Go to **Main Menu -> Window -> Preferences -> C/C++ -> Code Style -> Code Templates -> Files -> C++ Header File -> Default C++ header template**. Replace the template with the following:
::

    ${filecomment}

    #ifndef ${include_guard_symbol}
    #define ${include_guard_symbol}

    ${includes}

    ${namespace_begin}

    ${declarations}

    ${namespace_end}
    #endif  // ${include_guard_symbol}

Same goes to  Default C header template.

* Set the default "end of namespace declaration" template. Go to **Main Menu -> Window -> Preferences -> C/C++ -> Code Style -> Code Templates -> Code -> End of namespace declaration**. Edit the pattern to match
::

    }  // namespace ${namespace_name}


* Show the print margin. Got to **Main Menu -> Window -> Preferences -> General -> Editors -> Text Editors** and check "Show print margin".

*   Set default C++ source file format. Go to **Main Menu -> Window -> Preferences -> C/C++ -> Code Style -> Name Style -> Name Categories: -> Files/C++ Source File**.
    Set Capitalization to "Lower Case". Set Word Delimiter to "_". Set Suffix to ".cc". Preview should be "my_class.cc". Same goes to C++ Header File, but no suffix changed.

*   Load Google C++ Style Guide formatter for Eclipse. Go to **Main Menu -> Window -> Preferences -> C/C++ -> Code Style -> Formatter**, click **Import...**, set the path
    to previously downloaded official "Google C++ Style Guide formatter for Eclipse":http://google-styleguide.googlecode.com/svn/trunk/eclipse-cpp-google-style.xml.

* Setup pretty printing of C++ classes in GDB and Eclipse. Execute the following:
::

    sudo svn co svn://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/python /usr/share/libstdc++_printers
    sudo bash -c 'echo "# System-wide GDB initialization file.

    python
    import sys
    sys.path.insert(0, '"'"'/usr/share/libstdc++_printers'"'"')
    from libstdcxx.v6.printers import register_libstdcxx_printers
    register_libstdcxx_printers (None)
    end" > /etc/gdb/gdbinit'

Then go to **Main Menu -> Window -> Preferences -> C/C++ -> Debug -> GDB** and change ".gdbinit" to "/etc/gdb/gdbinit" (GDB command file). For any configuration
in **Main Menu -> Run -> Debug Configurations...** change **Debugger -> GDB command file** as well. If there is no "GDB" in the Preferences tree of vanilla Eclipse, you should try to debug some application first.
(Optional) Uncheck "Stop on startup at".

* (Optional) Set the Visual Studio key scheme. Go to **Main Menu -> Window -> Preferences -> General -> Keys** and select the "Microsoft Visual Studio" scheme.

* (Optional) Disable awkward adjusting indentation when pasting: Go to **Main Menu -> Window -> Preferences -> C/C++ -> Editor -> Typing** and uncheck "Adjust indentation".

* (Optional) Install "ANSI Console Eclipse plugin":http://www.mihai-nita.net/eclipse/ which supports colors.

* Activate pylint: <pre>sudo apt-get install pylint</pre>
Then go to **Main Menu -> Window -> Preferences -> PyDev -> PyLint**, check "Use PyLint?" and insert **/usr/bin/pylint** to "Location of the pylint executable". If you are lucky,
pylint will work out of the box then. Otherwise, PyDev will use a screwed PYTHONPATH and break pylint.

* PyDev Code Style: **Main Menu -> Window -> Preferences -> PyDev -> Editor -> Code Style -> Code Formatter**
You should set following checkboxes:

    * Auto-format editor contents before saving
    * Auto-format only files in the workspace
    * Use space after commas
    * Use space before and after operators (+,-,/,*,//,**,etc.)
    * Right trim lines
    * Right trim multi-line string literal
    * Add new line at the end of file
    * Spaces before a comment [2 spaces]
    * Spaces in comment start [At least 1 space]

* Create desktop launcher. Execute
::

    echo "[Desktop Entry]
    Comment=Eclipse CDT Kepler
    Terminal=false
    Name=Eclipse
    Exec=env UBUNTU_MENUPROXY= /var/eclipse/eclipse
    Type=Application
    Categories=IDE;Development
    X-Ayatana-Desktop-Shortcuts=NewWindow
    Icon=/var/eclipse/icon.xpm

    [NewWindow Shortcut Group]
    Name=New Window
    Exec=env UBUNTU_MENUPROXY= /var/eclipse/eclipse
    TargetEnvironment=Unity" > ~/.local/share/applications/eclipse.desktop


* The previous does not work (KGG)
Please add /etc/profile and /etc/environment the following line at start and
::

    UBUNTU_MENUPROXY=0

* THERE IS BETTER SOLUTION
::

    sudo apt-get autoremove appmenu-*
