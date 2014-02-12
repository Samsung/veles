# coding=utf-8
#
# fysom - pYthOn Finite State Machine - this is a port of Jake
#         Gordon's javascript-state-machine to python
#         https://github.com/jakesgordon/javascript-state-machine
#
# Copyright (C) 2011 Mansour Behabadi <mansour@oxplot.com>, Jake Gordon
#                                        and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__author__ = 'Mansour Behabadi'
__copyright__ = 'Copyright 2011, Mansour Behabadi and Jake Gordon'
__credits__ = ['Mansour Behabadi', 'Jake Gordon']
__license__ = 'MIT'
__version__ = '${version}'
__maintainer__ = 'Mansour Behabadi'
__email__ = 'mansour@oxplot.com'


WILDCARD = '*'


class FysomError(Exception):
    pass


class Fysom(object):

    def __init__(self, cfg):
        self._apply(cfg)

    def isstate(self, state):
        return self.current == state

    def can(self, event):
        return (event in self._map and ((self.current in self._map[event]) or WILDCARD in self._map[event])
                and not hasattr(self, 'transition'))

    def cannot(self, event):
        return not self.can(event)

    def is_finished(self):
        return self._final and (self.current == self._final)

    def _apply(self, cfg):
        init = cfg['initial'] if 'initial' in cfg else None
        if self._is_base_string(init):
            init = {'state': init}

        self._final = cfg['final'] if 'final' in cfg else None

        events = cfg['events'] if 'events' in cfg else []
        callbacks = cfg['callbacks'] if 'callbacks' in cfg else {}
        tmap = {}
        self._map = tmap

        def add(e):
            if 'src' in e:
                src = [e['src']] if self._is_base_string(
                    e['src']) else e['src']
            else:
                src = [WILDCARD]
            if e['name'] not in tmap:
                tmap[e['name']] = {}
            for s in src:
                tmap[e['name']][s] = e['dst']

        if init:
            if 'event' not in init:
                init['event'] = 'startup'
            add({'name': init['event'], 'src': 'none', 'dst': init['state']})

        for e in events:
            add(e)

        for name in tmap:
            setattr(self, name, self._build_event(name))

        for name in callbacks:
            setattr(self, name, callbacks[name])

        self.current = 'none'

        if init and 'defer' not in init:
            getattr(self, init['event'])()

    def _build_event(self, event):

        def fn(*args, **kwargs):

            if hasattr(self, 'transition'):
                raise FysomError(
                    "event %s inappropriate because previous transition did not complete" % event)
            if not self.can(event):
                raise FysomError(
                    "event %s inappropriate in current state %s" % (event, self.current))

            src = self.current
            dst = ((src in self._map[event] and self._map[event][src]) or
                   WILDCARD in self._map[event] and self._map[event][WILDCARD])

            class _e_obj(object):
                pass
            e = _e_obj()
            e.fsm, e.event, e.src, e.dst = self, event, src, dst
            for k in kwargs:
                setattr(e, k, kwargs[k])

            setattr(e, 'args', args)

            if self.current != dst:
                if self._before_event(e) is False:
                    return

                def _tran():
                    delattr(self, 'transition')
                    self.current = dst
                    self._enter_state(e)
                    self._change_state(e)
                    self._after_event(e)
                self.transition = _tran

            if self._leave_state(e) is not False:
                if hasattr(self, 'transition'):
                    self.transition()

        return fn

    def _before_event(self, e):
        fnname = 'onbefore' + e.event
        if hasattr(self, fnname):
            return getattr(self, fnname)(e)

    def _after_event(self, e):
        for fnname in ['onafter' + e.event, 'on' + e.event]:
            if hasattr(self, fnname):
                return getattr(self, fnname)(e)

    def _leave_state(self, e):
        fnname = 'onleave' + e.src
        if hasattr(self, fnname):
            return getattr(self, fnname)(e)

    def _enter_state(self, e):
        for fnname in ['onenter' + e.dst, 'on' + e.dst]:
            if hasattr(self, fnname):
                return getattr(self, fnname)(e)

    def _change_state(self, e):
        fnname = 'onchangestate'
        if hasattr(self, fnname):
            return getattr(self, fnname)(e)

    def _is_base_string(self, object):  # pragma: no cover
        try:
            return isinstance(object, basestring)
        except NameError:
            return isinstance(object, str)
