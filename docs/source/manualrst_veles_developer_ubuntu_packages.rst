:orphan:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install required Ubuntu packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    sudo apt-get install libgit2-dev libffi6 cython3 cython libhdf5-dev \
    unixodbc-dev liblzma-dev python3-pip python-pip python3-matplotlib \
    python-matplotlib libffi-dev python3-scipy python-scipy libsnappy-dev \
    python3-twisted python-twisted

Make sure, that there are no twisted in /usr/local/lib/python2.7/dist-packages/ and /usr/local/lib/python3.4/dist-packages/

Do not install twisted, matplotlib and scipy with pip or there could be problems.