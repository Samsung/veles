
Enable command line autocomplete
::::::::::::::::::::::::::::::::

Optionally, you can enable command line autocompletion for ``velescli.py``.
There should have already be ``argcomplete`` Python module installed via pip on your system,
since it is listed in Veles' requirements.txt. Execute the following::

    activate-global-python-argcomplete --dest=- >> ~/.bashrc
    
This command writes to your ``.bashrc`` file so that every time you open the
terminal argcomplete is activated (see `argcomplete description <https://pypi.python.org/pypi/argcomplete>`_).

.. note:: 
   Autocompletion does not work with the second way of running Veles (see :ref:`two_ways_of_running_veles`),
   that is, ``python3 -m veles``.