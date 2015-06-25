============
Snapshotting
============

:class:`veles.snapshotter.Snasphotter` unit allows dumping the current workflow
state in form of a `Python pickle <https://docs.python.org/3/library/pickle.html>`_.
Such "snapshot" can be stored on disk or in a database and used later to continue
the workflow's run from that point. In other words, snapshots are a kind of persistency.
To restore from a snapshot and continue running, ``--snapshot`` command-line
argument must be specified.

Internally, the main Veles workflow is pickled without the service classes such as
:class:`veles.launcher.Launcher`. This means that restoring from a snapshot requires
**the same workflow code as that the snapshot was taken with**, because
Python pickles contain only the data. Pickles are tolerant to code changes to some extent,
but best way is to backup the code of the workflow together with the snapshot.

Every unit is pickled; :class:`veles.distributable.Pickleable` base class ensures
that attributes ending with a low dash sign ("_") are not taken into the snapshot.
Such attributes should be created inside :func:`Unit.init_unpickled()` overriden
method, which is called after each restoration from a snapshot *and* when the unit
is created. Please note that :func:`__init__()` is **not** called after
unpickling. :func:`__getstate__` and :func:`__setstate__` methods are used, so
child classes must call :func:`super()` to override them.

Snapshots can be compressed with `Snappy <https://en.wikipedia.org/wiki/Snappy_(software)>`_,
Gzip, Bzip2 and Lzma2 (xz) algorithms. The default compression is Gzip. To change
the compression type, pass "compression" argument to :func:`__init__()`:


How to link snapshotters
::::::::::::::::::::::::

There is an important notice on how to link snapshotters. The should not be
any	uncertainty about other units' states at the point of taking of the snapshot,
because otherwise it will lead to hard to debug errors at restoration time. In
other words, one should link snapshotters in parallel with any other units except
the plotters. Moreover, a snapshotter must block the whole pipeline (be the part of a chain
without any bypassing links).

Snapshotting to a database
::::::::::::::::::::::::::

In case of
:class:`veles.znicz.standard_workflow.StandardWorkflow`, you must add "odbc" parameter
with `Open Database Connectivity <https://en.wikipedia.org/wiki/Open_Database_Connectivity>`_
datasource definition, e.g. ``DRIVER={MySQL};SERVER=localhost;DATABASE=test;UID=test;PWD=test``.
By default, "veles" table will be used; to change it, set "table" parameter.

In case of manual workflow construction, you must link
`veles.snapshotter.SnasphotterToDB` or any of it's children and pass "odbc" and
"table" constructor arguments.

The table scheme is as follows::

   +-----------+--------------+
   | Field     | Type         |
   +-----------+--------------+
   | timestamp | datetime     |
   | id        | char(36)     |
   | log_id    | char(36)     |
   | workflow  | varchar(100) |
   | name      | varchar(100) |
   | codec     | varchar(10)  |
   | data      | longblob     |
   +-----------+--------------+

