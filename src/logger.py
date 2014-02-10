"""
Created on Jul 12, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""
import logging


class Logger(object):
    """
    Provides logging facilities to derived classes.
    """

    def __init__(self):
        self.init_unpickled()

    def init_unpickled(self):
        self.logger_ = logging.getLogger(self.__class__.__name__)

    def log(self):
        """Returns the logger associated with this object.
        """
        return self.logger_


class Pickleable(Logger):
    """Will save attributes ending with _ as None when pickling and will call
    constructor upon unpickling.
    """
    def __init__(self):
        """Calls init_unpickled() to initialize the attributes which are not
        pickled.
        """
        super(Pickleable, self).__init__()
        # self.init_unpickled()  # already called in Logger()

    """This function is called if the object has just been unpickled.
    """
    def init_unpickled(self):
        if hasattr(super(Pickleable, self), "init_unpickled"):
            super(Pickleable, self).init_unpickled()

    def __getstate__(self):
        """Selects the attributes to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k[len(k) - 1] != "_" and not callable(v):
                state[k] = v
            else:
                state[k] = None
        return state

    def __setstate__(self, state):
        """Recovers the object after unpickling.
        """
        self.__dict__.update(state)
        self.init_unpickled()
