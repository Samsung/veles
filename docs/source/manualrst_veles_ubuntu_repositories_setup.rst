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

This step is needed only if smaug.rnd.samsung.ru is inaccessible from your site.
The easiest way to build a mirror is to install and use `aptly <http://www.aptly.info>`_.
Besides, you must obtain the  \*.deb files belonging to the repository from some other source,
e.g. via email or file sharing though 3-rd party. Execute and  carefully read the output::

    aptly repo create veles
    aptly repo add veles /path/to/deb/files
    aptly publish repo -distribution=trusty veles
    aptly serve

Replace "trusty" with the proper distribution codename (see ``lsb_release -c``).

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upgrading existing packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that your current packages are up to date::

    sudo apt-get update
    sudo apt-get upgrade
