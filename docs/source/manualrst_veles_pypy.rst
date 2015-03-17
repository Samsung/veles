Installing PyPy (Ubuntu)
========================

Add PyPy PPA and install PyPy::

    sudo add-apt-repository ppa:pypy/ppa
    sudo apt-get update
    sudo apt-get install pypy pypy-dev pypy-tk

Install pip::

    wget -O - https://bootstrap.pypa.io/get-pip.py | sudo -H pypy -
    sudo cp /usr/bin/{pip,pipy}
    sudo sed -i -e "s/python/pypy/" -e "s/==/>=/g" /usr/bin/pipy
    
Now you have a working pip for PyPy called pipy. Initialize the lazily
compiled packages (workaround for `#1999 <https://bitbucket.org/pypy/pypy/issue/1999/cffi-must-use-file-locks>`_)::

    sudo pypy -c "import sqlite3; import Tkinter"

Install numpy::

    sudo apt-get install libopenblas-dev liblapack-dev
    sudo -H pipy install git+https://bitbucket.org/pypy/numpy.git
    
Edit Veles' requirements.txt and comment (#) matplotlib and scipy. Finally::

   sudo -H pipy install -r requirements.txt backports.lzma
