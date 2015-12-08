User's setup (Cluster deployers and entry/medium level users of GNU/Linux)
==========================================================================

Cluster deployers and users of GNU/Linux other than Ubuntu should setup Veles
via ``deploy.sh`` script.

.. note:: If you are using Ubuntu, follow :doc:`manualrst_veles_ubuntu_user_setup`.

The first step depends on how one gets the Veles sources.

Via Git
:::::::

::

    sudo apt-get install git
    git clone https://github.com/Samsung/veles.git
    cd veles
    ./init

Execution of ``init`` script is neccessary, because Veles submodules require
special processing.
    
Via the redistributable package
:::::::::::::::::::::::::::::::

::

    mkdir Veles && cd Veles && tar -xf ../Veles.tar.xz
    
In case with Git, one can make his or her own Veles redistributable package
through running::

    deploy/deploy.sh pre
    

Prerequisites
:::::::::::::

.. include:: manualrst_veles_user_centos_packages.rst
.. include:: manualrst_veles_user_fedora_packages.rst

Bootstrapping
:::::::::::::

Execute the following script::

    deploy/deploy.sh post
    
It will build the needed environment, including Python interpreter, dependency
libraries, packages, etc. If everything is alright, proceed to the next step,
otherwise, please send an email with the script output (``deploy.sh post &> deploy.log``)
to gmarkhor@gmail.com.
 
The virtual environment will be located inside ``deploy/pyenv`` directory.
 
Going to the virtual environment
::::::::::::::::::::::::::::::::
To work with Veles, you will need to execute the following command every time
you open a new console session::

    cd <veles root>/deploy
    . ./init-pyenv
    
You may include these two lines into your ``.bashrc``.

PYTHONPATH
::::::::::

Optionally, you can add Veles root path to your PYTHONPATH for convenience.
Add ``export PYTHONPATH=$PYTHONPATH:<veles root path>`` to your ``.bashrc``.
This allows running ``python3 -m veles`` from any directory other than Veles
root.

Proceed to :doc:`manualrst_veles_ocl`.