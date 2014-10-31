:orphan:

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

Add the repository public key::

    wget -O - http://smaug.rnd.samsung.ru/apt/smaug.rnd.samsung.ru.gpg.key | sudo apt-key add -

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
