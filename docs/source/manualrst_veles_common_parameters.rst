:orphan:

Common parameters
:::::::::::::::::


You can see all common parameters at :mod:`veles.config`.
Common parameters you can change at ``root.common``.
You may change this common parameters:

.. code-block:: python

    root.common.update({

"matplotlib_backend" parameter allows to change Matplotlib Backend. The default
value is "Qt4Agg"

.. code-block:: python

        "matplotlib_backend": "Qt4Agg",

If "disable_plotting" parameter is True, graphics plotters will be disabled.
The default value is True for unit-tests and False otherwise.

.. code-block:: python

        "disable_plotting": "unittest" in sys.modules,

"precision_type" parameter is "float" or "double". The default value is "double"

.. code-block:: python

        "precision_type": "double",

"precision_level" parameter specified accuracy of calculation. 0 value is
for use simple summation. 1 value is for use Kahan summation (9% slower).
2 value is for use multipartials summation (90% slower). The default value is 0

.. code-block:: python

        "precision_level": 0,

"test_dataset_root" parameter sets the path to datasets directory

.. code-block:: python

        "test_dataset_root": os.path.join(os.environ.get("HOME", "./"), "data"),

If "disable_snapshots" parameter is True, Veles does not save workflow as
snapshot.

.. code-block:: python

        "disable_snapshots": False,

"engine.backend" parameter sets backend.  It could be "ocl", "cuda" or "auto".
The default value is "auto".

.. code-block:: python

        "engine": {"backend": "ocl"},

"cache_dir" parameter sets the path to cache directory where temporary files stored.
Plotters saving PDFs (killall -SIGUSR2 python3) :doc:`manualrst_veles_graphics`,
ImageSaver saving pictures and OpenCL caches code.

.. code-block:: python

        root.common.cache_dir = os.path.join(root.common.veles_user_dir, "cache")

"snapshot_dir" parameter sets the path to snapshots directory

.. code-block:: python

        root.common.snapshot_dir = os.path.join(
            root.common.veles_user_dir, "snapshots")
