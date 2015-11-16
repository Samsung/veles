User's setup on Ubuntu (level 1 and 2)
======================================

Users on Ubuntu have an ability to install Veles much easier than with other
methods.

Run command below::

    wget -O - https://velesnet.ml/ubuntu-install.sh | bash -

Or follow next steps

.. include:: manualrst_veles_ubuntu_repositories_setup.rst

Veles package installation
::::::::::::::::::::::::::

::

    sudo apt-get install python3-veles
    
If the installation failes on the last stage of fetching python dependecies via pip
with messages like::

     Download error on https://pypi.python.org/simple/pbr/: [Errno 110] Connection timed out -- Some packages may not be found!
     
most likely you hit `this issue <https://github.com/pypa/pip/issues/1805>`_ connected to the bug in "requests" Python package.
There are two options available then: install timed out packages by hand::

     sudo pip3 install <package name>
     
and then continue the installation::

     sudo apt-get install python3-veles
     
or use the workaround described in `this post <http://www.irvingc.com/posts/10>`_.