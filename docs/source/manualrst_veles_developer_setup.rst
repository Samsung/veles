Developer's setup on Ubuntu (medium/high level users)
=====================================================

The following procedure works only if your OS is Ubuntu and you have local
administrator's rights (that is, ``sudo``).

Besides, you are likely to need access to velesnet.ml.

Clone
:::::

::

    sudo apt-get install git
    git clone https://github.com/Samsung/veles.git
    cd veles
    ./init
    
Execution of ``init`` script is neccessary, because Veles submodules require
special processing.

.. include:: manualrst_veles_ubuntu_repositories_setup.rst

Packages installation
:::::::::::::::::::::

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upgrading existing packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that your current packages are up to date::

    sudo apt-get update
    sudo apt-get upgrade

.. include:: manualrst_veles_developer_ubuntu_packages.rst

^^^^^^^^^^^^^^
Python version
^^^^^^^^^^^^^^

Make sure you have only one instance of Python 3.x installed in system and it must be Python 3.4.1 or newer.
Run ``python3``. There should be printed ``Python 3.4.1 (default, Jul 26 2014, 16:51:18)`` or similar.
If not (can be with Ubuntu 14.04), double check that you successfully passed `Veles Debian repository`_ step.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install required pip packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Invoke pip install on requirements.txt. For python3::

   sudo pip3 install -r requirements.txt
   sudo pip3 install -r requirements-dev.3.txt

For python2::

   sudo pip install -r requirements.txt
   sudo pip install -r requirements-dev.2.txt

Looks like your Veles is ready to run! The last steps are :doc:`manualrst_veles_ocl`
and :doc:`manualrst_veles_eclipse` or :doc:`manualrst_veles_pycharm`.

^^^^^^^^^^^^^^^^^^^^^^^^
Set path to the datasets
^^^^^^^^^^^^^^^^^^^^^^^^

After first workflow run, you will see something like this::

    ERROR:MnistWorkflow:Unit "MnistLoader" failed to initialize
    ERROR:Launcher:Failed to initialize the workflow
    ERROR:Main:Failed to initialize the launcher.
    Traceback (most recent call last):
      File "/home/lyubov/Projects/veles/veles/__main__.py", line 639, in _main
        self.launcher.initialize(**kwargs)
      File "veles/launcher.py", line 391, in wrapped
        return fn(self, *args, **kwargs)
      File "veles/launcher.py", line 521, in initialize
        initialize_workflow()
      File "veles/launcher.py", line 517, in initialize_workflow
        raise from_none(ie)
    OSError: [Errno 2] No such file or directory: '/data/veles/datasets/MNIST'

It means that you need to change common datasets directory to yours. It could be done in veles/site_config.py.
Change "datasets" field in root.common.dirs from "/data/veles/datasets" to the preferred path with all datasets.
You don't need to download datasets in this folder, just make sure that this path exists and has correct permissions.

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

Installing veles-blog. How to create post
:::::::::::::::::::::::::::::::::::::::::

To create post in veles-blog (https://velesnet.ml/blog/ )::

    sudo pip3 install pelican typogrify

    sudo pip3 install git+https://github.com/vmarkovtsev/py-gfm.git

    git clone https://github.com/vmarkovtsev/veles-blog.git

    cd veles-blog

    make


Create post. Make sure, that it looks ok. Finally, change SSH_USER in Makefile and upload::

   make ssh_upload

Periodical system update
::::::::::::::::::::::::

In case of new packages and modules are released you can perform the update with
the following commands::

   sudo apt-get update && sudo apt-get upgrade
   sudo pip3 install -r requirements.txt --upgrade

