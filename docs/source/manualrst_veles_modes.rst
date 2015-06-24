:orphan:

Execution modes
:::::::::::::::

Veles can run workflows in different modes. Operation modes are:

* Standalone mode (default). Workflow runs on the same computer which Veles
  was invoked on.
* Master mode. Veles launches the server node for distributed computation. This
  involves passing ``-l / --listen-address`` command-line argument which specify
  the bind address and port. The actual communication will be done through
  `ZeroMQ <http://zeromq.org>`_ and the opened socket is used only for management.
* Slave mode. Veles launches the client node which does the actual calculations for
  it's master. The corresponding command-line argument is ``-m / --master-address``.

The modes above can be checked using :attr:`is_standalone`, :attr:`is_master` and
:attr:`is_slave` :class:`veles.units.Unit`'s properties. They are mutually exclusive.

Evaluation modes are:

* Train mode (default). The workflow is supposed to be trained.
* Test mode. Activated via ``--test``. The workflow is supposed to be run on
  unlabeled/untargeted data, thus loaders should fill only the TEST set. It is
  useless without setting ``--result-file`` argument (or writing the code to
  save the results by hand).

The modes above can be checked using :attr:`testing` :class:`veles.units.Unit`'s
property. They are mutually exclusive, too.

Overall, we get 3 * 2 = 6 different modes.