:orphan:

Repositories
::::::::::::

^^^^^^^^^^^^^^^^^^^^^^^
Veles Debian repository
^^^^^^^^^^^^^^^^^^^^^^^

Add Smaug repository to your /etc/apt/sources.list::

    sudo nano /etc/apt/sources.list

Add below line to the end of file::

    deb https://velesnet.ml/apt trusty main

On newer Ubuntu, replace trusty with your distribution code name.

If Smaug is not accessible from your point, there are two options left: either
fall back to :doc:`manualrst_veles_user_setup` or obtain a mirror server. The
latter requires you to have the archive with Smaug's repository beforehand and
``reprepro`` utility, see `Setting up a local mirror`_.

Add the repository public key::

    wget -O - https://velesnet.ml/apt/velesnet.ml.gpg.key | sudo apt-key add -
    
^^^^^^^^^^^
Proxy setup
^^^^^^^^^^^

This step is necessary **only if you use a proxy server** to connect to the internet.

If your apt uses a proxy server to connect to the internet (this is what we have
in Samsung headquaters), you must add the exclusion for ``velesnet.ml`` to
apt and make pip use your proxy. Add the following to ``/etc/apt/apt.conf``::

    Acquire::http::Proxy {
        velesnet.ml DIRECT;
    };
    
And add the following to ``~/.pip/pip.conf``::

    [global]
    proxy = http://168.219.61.252:8080
    no-check-certificate = True
    
**Create those configuration files as needed.** Replace ``168.219.61.252:8080``
with your proxy address. If your proxy is **not** an HTTP proxy (e.g., Tor),
pip setup will require some additional steps. ``proxy`` should be set to
``http://localhost:8123`` then.

If your pip's version is lower than 1.6 (true for Ubuntu <=14.10), you must
upgrade pip::

    GIT_SSL_NO_VERIFY=1 pip3 install git+https://github.com/djs/pip

"""""""""""""""""""""""""""""""""""""""""
Setting up HTTP to SOCKS proxy forwarding
"""""""""""""""""""""""""""""""""""""""""

We are going to use `polipo <http://www.pps.univ-paris-diderot.fr/~jch/software/polipo/>`_::

     sudo apt-get install polipo
    
Change the configuration file ``/etc/polipo/config``::

     proxyAddress = 127.0.0.1
     allowedClients = 127.0.0.1/32
     socksParentProxy = 168.219.61.252:8080
     
Restart the proxy server::

     sudo service polipo restart


^^^^^^^^^^^^^^^^^^^^^^^^^
Setting up a local mirror
^^^^^^^^^^^^^^^^^^^^^^^^^

This step is needed **only if velesnet.ml is inaccessible** from your site.
The easiest way to build a mirror is to install and use `aptly <http://www.aptly.info>`_.
Besides, you must obtain the  \*.deb files belonging to the repository from some other source,
e.g. via email or file sharing though 3-rd party. Execute and  carefully read the output::

    aptly repo create veles
    aptly repo add veles /path/to/deb/files
    aptly publish repo -distribution=trusty veles
    aptly serve

Replace "trusty" with the proper distribution codename (see ``lsb_release -c``).
