User's setup on Ubuntu (entry/medium level users)
=================================================

Users on Ubuntu have an ability to install Veles much easier than with other
methods.

Via ubintu-install script
:::::::::::::::::::::::::

Veles can be installed just with one command. Run::

    wget -O - https://velesnet.ml/ubuntu-install.sh | bash -

It will be located at /usr/lib/python3/dist-packages/veles.

After instalation of veles, copy samples from
/usr/lib/python3/dist-packages/veles/znicz/samples and /usr/lib/python3/dist-packages/veles/samples
to your local directory (/home/user/veles_samples for example). And run veles::

    python3 -m veles -s /home/user/veles_samples/MNIST/mnist.py -

If you have some issues, follow next steps

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