:orphan:

Using memory.Array
::::::::::::::::::

To simplify interacting with OpenCL devices, special class :class:`veles.memory.Array` was designed.

In order to use it:

1. Import the :class:`Array <veles.memory.Array>` class

.. code-block:: python

    from veles.memory import Array
    from veles.numpy_ext import roundup
    
  :func:`roundup() <veles.numpy_ext>` may be helpful for specifying ``ndrange`` when executing OpenCL kernels.
  
2. In constructor or in :meth:`initialize() <veles.opencl_units.OpenCLUnit>` create required vectors

.. code-block:: python

    self.a = Array(numpy.zeros([512, 1024], dtype=numpy.float32))
    self.b = Array()
    self.b.mem = numpy.zeros([1024, 1024], dtype=numpy.float32)
    
  Constructor of :class:`Array <veles.memory.Array>` may receive a numpy array, or numpy array can be assigned
  directly to the :attr:`mem <veles.memory.Array>` of the :class:`Array <veles.memory.Array>` instance.
  
3. Before supplying the vector to an OpenCL kernel
   (at the end of Unit's :meth:`initialize() <veles.opencl_units.OpenCLUnit>` when implementing the custom :doc:`manualrst_veles_units`)
   call :meth:`initialize() <veles.memory.Array>`

.. code-block:: python

    self.a.initialize(self)
    self.b.initialize(self, False)
    
  :meth:`initialize() <veles.memory.Array>` receives instance of :class:`OpenCLUnit <veles.opencl_units.OpenCLUnit>` as the first argument
  with optional argument ``bufpool`` (set it to False for vectors which have to live independently of the workflow,
  for example ``weights`` and ``bias`` but not ``input`` or ``output``).
  OpenCL buffer may or may not be created in :meth:`initialize() <veles.memory.Array>` depending on ``bufpool`` value.
  
4. Assign vectors to OpenCL kernels if necessary, this should be done usually in :meth:`ocl_init() <veles.opencl_units.OpenCLUnit>`

.. code-block:: python

    self.krn_.set_arg(0, self.a.devmem)
    self.krn_.set_arg(1, self.b.devmem)
    
  :attr:`devmem <veles.memory.Array>` is the OpenCL buffer handle.
  
5. Just before executing the OpenCL kernel call :meth:`unmap() <veles.memory.Array>` on the vectors it uses

.. code-block:: python

    self.a.unmap()
    self.b.unmap()
    self.execute_kernel(global_size, local_size, self.krn_)
    
  :meth:`unmap() <veles.memory.Array>` transfers data to OpenCL device from CPU address space only if it was mapped before,
  so it safe and fast to call it multiple times.
  
6. Before you want to use vector's data on the CPU, you have to call: :meth:`map_read() <veles.memory.Array>` and then use the data in read-only manner,
   :meth:`map_write() <veles.memory.Array>` and then update the data, :meth:`map_invalidate() <veles.memory.Array>` and then completely rewrite the data
   without caring for what it was in it before.
