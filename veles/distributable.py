"""
Created on May 26, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import functools
from six.moves import cPickle as pickle
import threading
from zope.interface import Interface, Attribute, implementer

import veles.logger as logger
from veles.mutable import LinkableAttribute


class Pickleable(logger.Logger):
    """Prevents attributes ending with _ from getting into pickle and calls
    init_unpickled() after unpickling to recover them.
    """
    def __init__(self, **kwargs):
        """Calls init_unpickled() to initialize the attributes which are not
        pickled.
        """
        self._method_storage = {}
        super(Pickleable, self).__init__(**kwargs)
        self.init_unpickled()

    def init_unpickled(self):
        """This function is called if the object has just been unpickled.
        """
        self.stripped_pickle_ = False
        self._pickle_lock_ = threading.Lock()
        for key, value in self._method_storage.items():
            class_method = getattr(value, key)
            setattr(self, key, functools.partial(class_method, self))

    def add_method_to_storage(self, name):
        """Convenience method to backup functions before wrapping them into
        decorators at runtime.
        """
        self._method_storage[name] = self.__class__

    def __getstate__(self):
        """Selects the attributes to pickle.
        """
        state = {}
        linked_values = {}
        for k, v in self.__dict__.items():
            if k[-1] != "_" and not callable(v):
                # Dereference the linked attributes in case of stripped pickle
                if (self.stripped_pickle and k[:2] == "__" and
                        isinstance(getattr(self.__class__, k[2:]),
                                   LinkableAttribute)):
                    linked_values[k[2:]] = getattr(self, k[2:])
                    state[k] = None
                else:
                    state[k] = v
            else:
                state[k] = None
        state.update(linked_values)

        # we have to check class attributes too
        # but we do not care of overriding (in __setstate__)
        if not self.stripped_pickle:
            class_attributes = {i: v
                                for i, v in self.__class__.__dict__.items()
                                if isinstance(v, LinkableAttribute)}
            state['class_attributes__'] = class_attributes
        return state

    def __setstate__(self, state):
        """Recovers the object after unpickling.
        """
        # recover class attributes
        if 'class_attributes__' in state:
            # RATS! AttributeError:
            # 'mappingproxy' object has no attribute 'update'
            # self.__class__.__dict__.update(state['class_attributes__'])
            for i, v in state['class_attributes__'].items():
                setattr(type(self), i, v)
            del state['class_attributes__']

        self.__dict__.update(state)
        super(Pickleable, self).__init__()
        self.init_unpickled()

    @property
    def stripped_pickle(self):
        return self.stripped_pickle_

    @stripped_pickle.setter
    def stripped_pickle(self, value):
        """A lock is taken if this is set to True and is released on False.
        """
        if value:
            self._pickle_lock_.acquire()
        self.stripped_pickle_ = value
        if not value:
            self._pickle_lock_.release()


class Distributable(Pickleable):
    DEADLOCK_TIME = 4

    def _data_threadsafe(self, fn, name):
        def wrapped(*args, **kwargs):
            if not self._data_lock_.acquire(
                    timeout=Distributable.DEADLOCK_TIME):
                self.error("Deadlock in %s: %s", self.name, name)
            else:
                self._data_lock_.release()
            with self._data_lock_:
                return fn(*args, **kwargs)

        return wrapped

    def __init__(self, **kwargs):
        self._generate_data_for_slave_threadsafe = \
            kwargs.get("generate_data_for_slave_threadsafe", True)
        self._apply_data_from_slave_threadsafe = \
            kwargs.get("apply_data_from_slave_threadsafe", True)
        super(Distributable, self).__init__(**kwargs)
        self.negotiates_on_connect = False
        self.add_method_to_storage("generate_data_for_slave")
        self.add_method_to_storage("apply_data_from_slave")

    def init_unpickled(self):
        super(Distributable, self).init_unpickled()
        self._data_lock_ = threading.Lock()
        self._data_event_ = threading.Event()
        self._data_event_.set()

        def make_threadsafe(name):
            func = getattr(self, name, None)
            if func is not None:
                setattr(self, name, self._data_threadsafe(func, name))

        if self._generate_data_for_slave_threadsafe:
            make_threadsafe("generate_data_for_slave")
        if self._apply_data_from_slave_threadsafe:
            make_threadsafe("apply_data_from_slave")
            make_threadsafe("drop_slave")

    @property
    def has_data_for_slave(self):
        return self._data_event_.is_set()

    @has_data_for_slave.setter
    def has_data_for_slave(self, value):
        if value:
            self.debug("%s has data for slave", self.name)
            self._data_event_.set()
        else:
            self.debug("%s has NO data for slave", self.name)
            self._data_event_.clear()

    def wait_for_data_for_slave(self):
        if not self._data_event_.wait(Distributable.DEADLOCK_TIME):
            self.error("Deadlock in %s: wait_for_data_for_slave", self.name)
            self._data_event_.wait()

    def save(self, file_name):
        """
        Stores object's current state in the specified file.
        """
        data = self.generate_data_for_slave()
        pickle.dump(data, file_name)

    def load(self, file_name):
        """
        Loads object's current state from the specified file.
        """
        data = pickle.load(file_name)
        self.apply_data_from_master(data)


class IDistributable(Interface):
    """Classes which provide this interface can be used in distributed
    computation environments.
    """

    negotiates_on_connect = Attribute("""Flag indicating whether """
        """generate() and apply() must be called during the initial """
        """connect phase.""")

    def generate_data_for_master():
        """Data for master should be generated here. This function is executed
        on a slave instance.

        Returns:
            data of any type or None if there is nothing to send.
        """

    def generate_data_for_slave(slave):
        """Data for slave should be generated here. This function is executed
        on a master instance.
        This method is guaranteed to be threadsafe if
        generate_data_for_slave_threadsafe is set to True in __init__.

        Parameters:
            slave: some information about the slave (may be None).

        Returns:
            data of any type or None if there is nothing to send.
        """

    def apply_data_from_master(data):
        """Data from master should be applied here. This function is executed
        on a slave instance.

        Parameters:
            data - exactly the same value that was returned by
                   generate_data_for_slave at the master's side.

        Returns:
            None.
        """

    def apply_data_from_slave(data, slave):
        """Data from slave should be applied here. This function is executed
        on a master instance.
        This method is guaranteed to be threadsafe if
        apply_data_from_slave_threadsafe is set to True in __init__ (default).

        Parameters:
            slave: some information about the slave (may be None).

        Returns:
            None.
        """

    def drop_slave(slave):
        """Unexpected slave disconnection leads to this function call.
        This method is guaranteed to be threadsafe if
        apply_data_from_slave_threadsafe is set to True in __init__ (default).
        """


@implementer(IDistributable)
class TriviallyDistributable(object):
    """Empty IDistributable implementation for special units.
    """

    def generate_data_for_master(self):
        return None

    def generate_data_for_slave(self, slave):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass
