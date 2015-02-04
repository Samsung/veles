"""
Created on May 15, 2014

Enables the interactive debugging of errors occured during pickling
and unpickling.

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import six
import pickle
import sys
from pickle import PicklingError, UnpicklingError
import warnings


# : The best protocol value for pickle().
best_protocol = 4 if sys.version_info > (3, 4) else sys.version_info[0]


def setup_pickle_debug():
    """Enables the interactive debugging of errors occured during pickling
    and unpickling.
    """
    if not six.PY3:
        warnings.warn("Pickle debugging is only available for Python 3.x")
        return

    pickle.dump = pickle._dump
    pickle.dumps = pickle._dumps
    pickle.load = pickle._load
    pickle.loads = pickle._loads
    orig_save = pickle._Pickler.save
    orig_load = pickle._Unpickler.load

    def save(self, obj):
        try:
            orig_save(self, obj)
        except (PicklingError, TypeError, AssertionError):
            import traceback
            import pdb
            six.print_("\033[1;31mPickling failure\033[0m", file=sys.stderr)
            traceback.print_exc()
            # look at obj
            pdb.set_trace()

    def load(self):
        try:
            orig_load(self)
        except (UnpicklingError, ImportError, AssertionError):
            import traceback
            import pdb
            six.print_("\033[1;31mUnpickling failure\033[0m", file=sys.stderr)
            traceback.print_exc()
            # examine the exception
            pdb.post_mortem()

    pickle._Pickler.save = save
    pickle._Unpickler.load = load
