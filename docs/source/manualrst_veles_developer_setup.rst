Developer's setup (level 2 and 3)
=================================

The following procedure works only if your OS is Ubuntu and you have local
administrator's rights (that is, ``sudo``).

Besides, you are likely to need access to smaug.rnd.samsung.ru.

Clone
:::::

::

    git clone ssh://<user name>@alserver.rnd.samsung.ru:29418/Veles
    cd Veles
    ./init
    
If you don't have a user name, please contact the administator: v.markovtsev@samsung.com.
Execution of ``init`` script is neccessary, because Veles submodules require
special processing.

Repositories
::::::::::::

^^^^^^^^^^^^^^^^^^^^^^^
Veles Debian repository
^^^^^^^^^^^^^^^^^^^^^^^

Add Smaug repository to your /etc/apt/sources.list: ``deb http://smaug.rnd.samsung.ru/apt trusty main``
On newer Ubuntu, replace trusty with your distribution code name.

If Smaug is not accessible from your point, there are two options left: either
fall back to :doc:`manualrst_veles_user_setup` or obtain a mirror server. The
latter requires you to have the archive with Smaug's repository beforehand and
``reprepro`` utility, see `Setting up a local mirror`_.

^^^^^^^^^^^^^^^^^^^^^^^^^
Setting up a local mirror
^^^^^^^^^^^^^^^^^^^^^^^^^

This step is neede only if smaug.rnd.samsung.ru is inaccessible. Suppose that
you received the archive with the repository named, e.g., "veles-repo.tar.xz".
At first, make sure that you have ``reprepro`` installed. Execute the following::

    mkdir veles-repo && cd veles-repo && tar -xf ../veles-repo.tar.xz
 
Then you need to setp a new GnuPG key which is used to sign all packages,
Please refer to `Debian wiki <https://wiki.debian.org/SettingUpSignedAptRepositoryWithReprepro>`_.
Finally, add the packages::

    sudo reprepro -Vb . includedeb trusty deb/*
    
Depending on which web server you are running, you will have to add the corresponding
proxy rule. Here is the nginx's rule on Smaug::

    location ~ "^/apt/(.*)$" {
        alias /data/veles/packages/binary/$1;
    }


^^^^^^^^^^^^^^^^^^^^^
Matplotlib repository
^^^^^^^^^^^^^^^^^^^^^

If your matplotlib's version is older than 1.4 (this is likely to be on Ubuntu
14.04), you should add this private Matplotlib ppa: ``ppa:takluyver/matplotlib-daily``::

    sudo add-apt-repository ppa:takluyver/matplotlib-daily


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upgrading existing packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that your current packages are up to date::

    sudo apt-get update
    sudo apt-get upgrade

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