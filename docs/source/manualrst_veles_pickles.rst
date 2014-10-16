:orphan:

Pickling in Veles
:::::::::::::::::

:mod:`pickle2 <veles.pickle2>` module was specially designed to easen proper
pickling and unpickling throught Veles source code.

It defines :const:`best_protocol <veles.pickle2.best_protocol>` which should be passed
into :func:`pickle.dump()` as :attr:`protocol` value. The reason of this constant is
to make :func:`pickle.dump()` use protocol version 4 on Python 3.4 (it uses 3 by default).

:func:`setup_pickle_debug() <veles.pickle2.setup_pickle_debug>` patches
:func:`pickle.dump()`, :func:`pickle.dumps()`, :func:`pickle.load()` and
:func:`pickle.loads()` to use Python implementation instead of the native one,
so that pickling/unpickling errors are printed with nice stack traces,
the guilty object is revealed and ``pdb`` session is triggered.

.. note:: 
   Pickles debugging can be activated via ``--debug-pickle`` command line option.