"""
Created on May 15, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import six
from six.moves import cPickle as pickle
from pickle import PicklingError, UnpicklingError
import warnings


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
        except PicklingError:
            import traceback
            import pdb
            print("\033[1;31mPickling failure\033[0m")
            traceback.print_exc()
            # look at obj
            pdb.set_trace()

    def load(self):
        try:
            orig_load(self)
        except (UnpicklingError, ImportError):
            import traceback
            import pdb
            print("\033[1;31mUnpickling failure\033[0m")
            traceback.print_exc()
            # examine the exception
            pdb.post_mortem()

    pickle._Pickler.save = save
    pickle._Unpickler.load = load
