# -*- coding: utf-8 -*-
#
# Copyright (c) 2012 Red Hat, Inc.
#
# kitchen is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# kitchen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with kitchen; if not, see <http://www.gnu.org/licenses/>
#
# Authors:
#   Toshio Kuratomi <toshio@fedoraproject.org>
#   Seth Vidal
#
# Portions of code taken from yum/i18n.py and
# python-fedora: fedora/textutils.py


import codecs
import warnings


def isunicodestring(obj):
    '''Determine if obj is a :class:`unicode` string
    In python2 this is equivalent to isinstance(obj, unicode).  In python3 it
    checks whether the object is an instance of :class:`str`.
    :arg obj: Object to test
    :returns: True if the object is a :class:`unicode` string.  Otherwise, False.
    .. versionadded:: Kitchen: 1.2.0, API kitchen.text 2.2.0
    '''
    if isinstance(obj, unicode):
        return True
    return False


def to_bytes(obj, encoding='utf-8', errors='replace', nonstring=None,
             non_string=None):
    '''Convert an object into a byte :class:`str`
    :arg obj: Object to convert to a byte :class:`str`.  This should normally
        be a :class:`unicode` string.
    :kwarg encoding: Encoding to use to convert the :class:`unicode` string
        into a byte :class:`str`.  Defaults to :term:`utf-8`.
    :kwarg errors: If errors are found while encoding, perform this action.
        Defaults to ``replace`` which replaces the invalid bytes with
        a character that means the bytes were unable to be encoded.  Other
        values are the same as the error handling schemes in the `codec base
        classes
        <http://docs.python.org/library/codecs.html#codec-base-classes>`_.
        For instance ``strict`` which raises an exception and ``ignore`` which
        simply omits the non-encodable characters.
    :kwarg nonstring: How to treat nonstring values.  Possible values are:
        :simplerepr: Attempt to call the object's "simple representation"
            method and return that value.  Python-2.3+ has two methods that
            try to return a simple representation: :meth:`object.__unicode__`
            and :meth:`object.__str__`.  We first try to get a usable value
            from :meth:`object.__str__`.  If that fails we try the same
            with :meth:`object.__unicode__`.
        :empty: Return an empty byte :class:`str`
        :strict: Raise a :exc:`TypeError`
        :passthru: Return the object unchanged
        :repr: Attempt to return a byte :class:`str` of the :func:`repr` of the
            object
        Default is ``simplerepr``.
    :kwarg non_string: *Deprecated* Use :attr:`nonstring` instead.
    :raises TypeError: if :attr:`nonstring` is ``strict`` and
        a non-:class:`basestring` object is passed in or if :attr:`nonstring`
        is set to an unknown value.
    :raises UnicodeEncodeError: if :attr:`errors` is ``strict`` and all of the
        bytes of :attr:`obj` are unable to be encoded using :attr:`encoding`.
    :returns: byte :class:`str` or the original object depending on the value
        of :attr:`nonstring`.
    .. warning::
        If you pass a byte :class:`str` into this function the byte
        :class:`str` is returned unmodified.  It is **not** re-encoded with
        the specified :attr:`encoding`.  The easiest way to achieve that is::
            to_bytes(to_unicode(text), encoding='utf-8')
        The initial :func:`to_unicode` call will ensure text is
        a :class:`unicode` string.  Then, :func:`to_bytes` will turn that into
        a byte :class:`str` with the specified encoding.
    Usually, this should be used on a :class:`unicode` string but it can take
    either a byte :class:`str` or a :class:`unicode` string intelligently.
    Nonstring objects are handled in different ways depending on the setting
    of the :attr:`nonstring` parameter.
    The default values of this function are set so as to always return a byte
    :class:`str` and never raise an error when converting from unicode to
    bytes.  However, when you do not pass an encoding that can validly encode
    the object (or a non-string object), you may end up with output that you
    don't expect.  Be sure you understand the requirements of your data, not
    just ignore errors by passing it through this function.
    .. versionchanged:: 0.2.1a2
        Deprecated :attr:`non_string` in favor of :attr:`nonstring` parameter
        and changed default value to ``simplerepr``
    '''
    # Could use isbasestring, isbytestring here but we want this to be as fast
    # as possible
    if isinstance(obj, basestring):
        if isinstance(obj, str):
            return obj
        return obj.encode(encoding, errors)
    if non_string:
        warnings.warn('non_string is a deprecated parameter of'
                      ' to_bytes().  Use nonstring instead',
                      DeprecationWarning,
                      stacklevel=2)
        if not nonstring:
            nonstring = non_string
    if not nonstring:
        nonstring = 'simplerepr'

    if nonstring == 'empty':
        return ''
    elif nonstring == 'passthru':
        return obj
    elif nonstring == 'simplerepr':
        try:
            simple = str(obj)
        except UnicodeError:
            try:
                simple = obj.__str__()
            except (AttributeError, UnicodeError):
                simple = None
        if not simple:
            try:
                simple = obj.__unicode__()
            except (AttributeError, UnicodeError):
                simple = ''
        if isunicodestring(simple):
            simple = simple.encode(encoding, 'replace')
        return simple
    elif nonstring in ('repr', 'strict'):
        try:
            obj_repr = obj.__repr__()
        except (AttributeError, UnicodeError):
            obj_repr = ''
        if isunicodestring(obj_repr):
            obj_repr = obj_repr.encode(encoding, errors)
        else:
            obj_repr = str(obj_repr)
        if nonstring == 'repr':
            return obj_repr
        raise TypeError('to_bytes was given "%(obj)s" which is neither'
                        ' a unicode string or a byte string (str)' % {
            'obj': obj_repr})

    raise TypeError('nonstring value, %(param)s, is not set to a valid'
                    ' action' % {'param': nonstring})


def getwriter(encoding):
    '''Return a :class:`codecs.StreamWriter` that resists tracing back.
    :arg encoding: Encoding to use for transforming :class:`unicode` strings
        into byte :class:`str`.
    :rtype: :class:`codecs.StreamWriter`
    :returns: :class:`~codecs.StreamWriter` that you can instantiate to wrap output
        streams to automatically translate :class:`unicode` strings into :attr:`encoding`.
    This is a reimplemetation of :func:`codecs.getwriter` that returns
    a :class:`~codecs.StreamWriter` that resists issuing tracebacks.  The
    :class:`~codecs.StreamWriter` that is returned uses
    :func:`kitchen.text.converters.to_bytes` to convert :class:`unicode`
    strings into byte :class:`str`.  The departures from
    :func:`codecs.getwriter` are:
    1) The :class:`~codecs.StreamWriter` that is returned will take byte
       :class:`str` as well as :class:`unicode` strings.  Any byte
       :class:`str` will be passed through unmodified.
    2) The default error handler for unknown bytes is to ``replace`` the bytes
       with the unknown character (``?`` in most ascii-based encodings, ``�``
       in the utf encodings) whereas :func:`codecs.getwriter` defaults to
       ``strict``.  Like :class:`codecs.StreamWriter`, the returned
       :class:`~codecs.StreamWriter` can have its error handler changed in
       code by setting ``stream.errors = 'new_handler_name'``
    Example usage::
        $ LC_ALL=C python
        >>> import sys
        >>> from kitchen.text.converters import getwriter
        >>> UTF8Writer = getwriter('utf-8')
        >>> unwrapped_stdout = sys.stdout
        >>> sys.stdout = UTF8Writer(unwrapped_stdout)
        >>> print 'caf\\xc3\\xa9'
        café
        >>> print u'caf\\xe9'
        café
        >>> ASCIIWriter = getwriter('ascii')
        >>> sys.stdout = ASCIIWriter(unwrapped_stdout)
        >>> print 'caf\\xc3\\xa9'
        café
        >>> print u'caf\\xe9'
        caf?
    .. seealso::
        API docs for :class:`codecs.StreamWriter` and :func:`codecs.getwriter`
        and `Print Fails <http://wiki.python.org/moin/PrintFails>`_ on the
        python wiki.
    .. versionadded:: kitchen 0.2a2, API: kitchen.text 1.1.0
    '''

    class _StreamWriter(codecs.StreamWriter):
        # :W0223: We don't need to implement all methods of StreamWriter.
        #   This is not the actual class that gets used but a replacement for
        #   the actual class.
        # :C0111: We're implementing an API from the stdlib.  Just point
        #   people at that documentation instead of writing docstrings here.
        # pylint:disable-msg=W0223,C0111
        def __init__(self, stream, errors='replace'):
            codecs.StreamWriter.__init__(self, stream, errors)

        def encode(self, msg, errors='replace'):
            return (to_bytes(msg, encoding=self.encoding, errors=errors),
                    len(msg))

    _StreamWriter.encoding = encoding
    return _StreamWriter
