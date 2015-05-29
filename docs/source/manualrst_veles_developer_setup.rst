Developer's setup (level 2 and 3)
=================================

The following procedure works only if your OS is Ubuntu and you have local
administrator's rights (that is, ``sudo``).

Besides, you are likely to need access to velesnet.ml.

Clone
:::::

::

    git clone ssh://<user name>@alserver.rnd.samsung.ru:29418/Veles
    cd Veles
    ./init
    
If you don't have a user name, please contact the administator: v.markovtsev@samsung.com.
Execution of ``init`` script is neccessary, because Veles submodules require
special processing.

.. include:: manualrst_veles_ubuntu_repositories_setup.rst

Packages installation
:::::::::::::::::::::

.. include:: manualrst_veles_developer_ubuntu_packages.rst

Check that the patched twisted and pyzmq are installed::

   python3
   >>> import twisted.web.client
   >>> import zmq
   
No error messages should appear!

^^^^^^^^^^^^^^
Python version
^^^^^^^^^^^^^^

Make sure you have only one instance of Python 3.x installed in system and it must be Python 3.4.1 or newer.
Run ``python3``. There should be printed ``Python 3.4.1 (default, Jul 26 2014, 16:51:18)`` or similar.
If not (can be with Ubuntu 14.04), double check that you successfully passed `Veles Debian repository`_ step.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install required pip packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Invoke pip install on requirements.txt::

   sudo pip3 install -r requirements.txt
   
Looks like your Veles is ready to run! The last steps are :doc:`manualrst_veles_ocl`
and :doc:`manualrst_veles_eclipse` or :doc:`manualrst_veles_pycharm`.

Importing project
:::::::::::::::::

^^^^^^^
Eclipse
^^^^^^^

Select **Main Menu -> File -> Import...**. Click
**General/Existing Projects into Workspace**. Click "Browse" to the right
of root directory and specify the directory with Veles. Click "Finish".

^^^^^^^
PyCharm
^^^^^^^

Select **Open Directory** and specify the directory with Veles.

Installing Sphinx (generator of this document)
::::::::::::::::::::::::::::::::::::::::::::::

Optionally, you can install documentation build support.
Make sure that "python2" version of Sphinx is NOT installed::

    sudo apt-get remove python-sphinx

Then install "python3" Sphinx::

    sudo apt-get install python3-sphinx3

and ensure all Sphinx plugins are installed::

    sudo pip3 install -r docs/requirements.txt

Go to the `docs` folder and try to generate the documentation::

   ./generate_docs.py

Output files will be located in docs/build/html.

Periodical system update
::::::::::::::::::::::::

In case of new packages and modules are released you can perform the update with
the following commands::

   sudo apt-get update && sudo apt-get upgrade
   sudo pip3 install -r requirements.txt --upgrade
   
.. include:: manualrst_veles_cmdline_autocomplete.rst