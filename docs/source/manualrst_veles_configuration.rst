:orphan:

=============
Configuration
=============

Config class
::::::::::::

Veles has a global tree of all settings which resides in :mod:`veles.config`.
The root node is ``veles.config.root``. All settings which are not specific to
the loaded workflow are children of ``root.common``, otherwise each workflow
has it's own namespace inside ``root``, e.g. ``root.mnist_all2all`` (see
:doc:`manualrst_veles_using_configs`). All tree nodes except leaves are
instances of :class:`veles.config.Config`.

``Config`` class feels like a Javascript object, that is, it's attributes are
either ``Config``-s or ordinary Python objects, like strings or lists. New
children are added lazily, that is, if a ``Config`` instance "cfg" does not have an
attribute "file", ``value = cfg.file`` will assign a new plain ``Config`` to
"value", while ``cfg.file = "path"`` will set the attribute value to "path".
This allows us to write ``cfg.child.other.sub.threshold = 10`` without ensuring
that all intermediate nodes exist.

``Config`` objects can be :func:`update()`-d. :func:`update()` method takes
a dictionary or a ``Config`` and reconstructs the tree according to it. For
example,

   .. code-block:: python

      cfg.update({
          "one": {
              "two": 2,
              "three": {
                  "four": 4
              }
          }
      })

creates ``cfg.one.two`` which equals to 2, ``cfg.one.three.four`` which equals to 4.
If any nodes already exist, their values are overwritten.
The code above looks like JSON and feels like JSON and was designed to be like JSON.

``Config`` objects can be saved to JSON. One has to specify the custom encoder class:

   .. code-block:: python

   import json
   from veles.json_encoders import ConfigJSONEncoder
   from veles.config import root

   print(json.dumps(root, cls=ConfigJSONEncoder))

``Config`` objects are pickleable as well.

``Config`` objects can be pretty-printed using :func:`print_()`.

Configuration at import time
::::::::::::::::::::::::::::

While ``veles.config`` is being imported, Veles searches and applies configuration updates
from the following sources, skipping non-existent paths:

#. Default root configuration inside ``veles.config``
#. ``veles.site_config``
#. ``root.common.dirs.dist_config`` / ``site_config.py``
#. ``root.common.dirs.user`` / ``site_config.py``
#. Current working directory / ``site_config.py``

``site_config.py`` should contain the function :func:`update()` which takes
``root`` as a single argument and changes it inside. By default,
``root.common.dirs.dist_config`` points to "/etc/default/veles" and
``root.common.dirs.user`` points to current user's home directory with ".veles"
appended. Thus it is possible to change Veles' start settings in many ways.
