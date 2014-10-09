User's setup (level 1)
======================

Normal users should setup Veles via ``deploy.sh`` script. This method should
also work on systems other than Ubuntu. The first step depends on how one gets
the Veles sources.

Via Git
:::::::

::

    git clone http://alserver.rnd.samsung.ru/gerrit/Veles
    cd Veles
    ./init

Execution of ``init`` script is neccessary, because Veles submodules require
special processing. Substitute ``alserver.rnd.samsung.ru/gerrit`` with your
local mirror, if neccessary.
    
Via the redistributable package
:::::::::::::::::::::::::::::::

::

    mkdir Veles && cd Veles && tar -xf ../Veles.tar.xz
    
In case with Git, one can make his or her own Veles redistributable package
through running::

    deploy/deploy.sh pre
    

Bootstrapping
:::::::::::::

Execute the following script::

    deploy/deploy.sh post
    
It will build the needed environment, including Python interpreter, dependency
libraries, packages, etc. If everything is alright, proceed to the next step,
otherwise, please send an email with the script output (``deploy.sh post &> deploy.log``)
to v.markovtsev@samsung.com.    
 
The virtual environment will be located inside ``deploy/pyenv`` directory.
 
Going to the virtual environment
::::::::::::::::::::::::::::::::
To work with Veles, you will need to execute the following command every time
you open a new console session::

    deploy/init-pyenv
    pyenv local 3.4.1
    
You may include these two lines into your ``.bashrc``. "3.4.1" is the version
of Python interpreter installed into the virtual environment and may change in
the future.

PYTHONPATH
::::::::::

Optionally, you can add Veles root path to your PYTHONPATH for convenience.
Add ``export PYTHONPATH=$PYTHONPATH:<veles root path>`` to your ``.bashrc``.
This allows running ``python3 -m veles`` from any directory other than Veles
root.

Proceed to :doc:`manualrst_veles_ocl`.