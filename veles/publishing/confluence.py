#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Copyright (c) 2015, Samsung Electronics Co.,Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of Samsung Electronics Co.,Ltd.
"""


from __future__ import print_function
from mimetypes import guess_type
import os
from six.moves import xmlrpc_client as xmlrpclib
import socket
import ssl

from veles.logger import Logger


class Confluence(Logger):
    def __init__(self, server, username, password, timeout=120):
        """
        Initializes a new Confluence XML RPC API client instance.

        :param server: URL of the Confluence server.
        :param username: username to use for authentication.
        :param password: password to use for authentication.

        Usage:

            >>> from confluence import Confluence
            >>>
            >>> conf = Confluence("http://localhost:8080", "admin", "admin")
            >>> conf.store_page_content("test", "test", "hello world!")

        """
        super(Confluence, self).__init__()
        # without this there is no timeout, and this may block the requests
        socket.setdefaulttimeout(timeout)

        self._server = xmlrpclib.ServerProxy(
            server + '/rpc/xmlrpc', allow_none=True)

        for version in "confluence2", "confluence1":
            try:
                self._token = getattr(self._server, version).login(
                    username, password)
                self.info("Logged in via \"%s\" as \"%s\"", version, username)
                self._version = version
                self._api = getattr(self._server, version)
                break
            except xmlrpclib.Error as e:
                self.debug("Failed \"%s\" login: %s", version, e)
        else:
            raise ValueError(
                "Could not login to %s as %s" % (server, username))

    @property
    def token(self):
        return self._token

    @property
    def server(self):
        return self._server

    @property
    def version(self):
        return self._version

    def unsupported_in_api_version_1(fn):
        def wrapped_unsupported_in_api_version_1(self, *args, **kwargs):
            assert self.version != "confluence1"
            return fn(self, *args, **kwargs)
        return wrapped_unsupported_in_api_version_1

    def get_page(self, page, space):
        """
        Returns a page object as a dictionary. If it does not exist, raises
        xmlrpclib.Fault.

        :param page: Page title.
        :param space: The space *key* (usually, 3 capital letters).
        :return: dictionary. result['content'] contains the body of the page.
        """
        return self._api.getPage(self.token, space, page)

    def get_page_summary(self, page, space):
        """
        Returns a page object as a dictionary. If it does not exist, returns
        None.

        :param page: Page title.
        :param space: The space *key* (usually, 3 capital letters).
        """
        try:
            return self._api.getPageSummary(self.token, space, page)
        except xmlrpclib.Fault:
            return None

    def attach_file(self, page, space, files, comments=None):
        existing_page = self.get_page(page, space)
        if comments is None:
            comments = {}
        for file_name, data in files.items():
            try:
                self._api.removeAttachment(self.token, existing_page["id"],
                                           file_name)
            except xmlrpclib.Fault:
                self.debug("No existing attachment to replace")
            mime = guess_type(file_name)[0]
            if mime is None:
                self.warning("Could not determine MIME type of %s", file_name)
                mime = "application/binary"
            attachment = {
                "fileName": file_name, "contentType": mime,
                "comment": comments.get(file_name)}
            if isinstance(data, str) and os.path.exists(data):
                with open(data, "rb") as fin:
                    data = fin.read()
            if not isinstance(data, bytes):
                raise TypeError("%s has the wrong data type \"%s\"" %
                                (file_name, type(data)))
            try:
                self.debug("Calling addAttachment(%s, %s, %s, ...)",
                           self.token, existing_page["id"], attachment)
                self._api.addAttachment(
                    self.token, existing_page["id"], attachment,
                    xmlrpclib.Binary(data))
                self.debug("Uploaded %s", file_name)
            except xmlrpclib.Error:
                self.exception("Unable to attach %s", file_name)

    def remove_all_attachments(self, page, space):
        """
        Removes all the attachments for the specified page.
        :param page: Page title.
        :param space: The space *key* (usually, 3 capital letters).
        :return: None.
        """
        existing_page = self.get_page(page, space)

        # Get a list of attachments
        files = self._api.getAttachments(self.token, existing_page["id"])

        # Iterate through them all, removing each
        numfiles = len(files)
        for i, f in enumerate(files):
            filename = f['fileName']
            self.debug("Removing %d of %d (%s)..." % (i, numfiles, filename))
            self._api.removeAttachment(
                self.token, existing_page["id"], filename)

    def get_blog_entries(self, space):
        """
        Returns a page object as a Array.

        :param space: The space *key* (usually, 3 capital letters).
        """
        return self._api.getBlogEntries(self.token, space)

    def get_blog_entry(self, page_id):
        """
        Returns a blog page as a BlogEntry object.

        :param page_id: Page identifier.
        """
        if self.version == "confluence2":
            return self._api.getBlogEntry(self.token, page_id)
        return self._api.getBlogEntries(self._token, page_id)

    def store_blog_entry(self, entry):
        """
        Store or update blog content.
        (The BlogEntry given as an argument should have space, title and
        content fields at a minimum.)

        :param entry:
        :return: blogEntry: if succeeded
        """
        return self._api.storeBlogEntry(self.token, entry)

    def add_label_by_name(self, label_name, object_id):
        """
        Adds label(s) to the object.

        :param label_name (Tag Name)
        :param object_id (Such as pageId)
        :retuen: bool: True if succeeded
        """
        return self._api.addLabelByName(self.token, label_name, object_id)

    def get_page_id(self, page, space):
        """
        Retuns the numeric id of a confluence page.

        :param page: Page title.
        :param space: The space *key* (usually, 3 capital letters).
        :return: Integer: page numeric id
        """
        return self.get_page(page, space)["id"]

    def store_page_content(self, page, space, content, convert_wiki=True,
                           parent=None):
        """
        Modifies the content of a Confluence page.

        :param page: Page title.
        :param space: The space *key* (usually, 3 capital letters).
        :param content: The new content of the page.
        :param convert_wiki: With API version 1, this is ignored. With API
                             version 2, specifies the value indicating whether
                             to perform Confluence wiki markup conversion for
                             the content. Thus, False means the content is
                             supplied in raw HTML format.
        :return: bool: True if succeeded
        """
        if self.version == "confluence2" and convert_wiki:
            content = self.convert_wiki_to_storage_format(content)
            self.debug("Converted content: %s", content)
        try:
            data = self.get_page(page, space)
            data['content'] = content
        except xmlrpclib.Fault:
            data = {"title": page, "content": content, "space": space,
                    "parentId": self.get_page_id(parent, space)
                    if parent is not None else "0"}
        return self._api.storePage(self.token, data)

    def render_content(self, page, space, a='', b=None):
        """
        Obtains the HTML content of a wiki page.

        :param space: The space *key* (usually, 3 capital letters).
        :param page: Page title.
        :return: string: HTML content
        """
        try:
            if not page.isdigit():  # isinstance(page, numbers.Integral):
                page = self.get_page_id(page=page, space=space)
            return self._api.renderContent(self.token, space, page, a, b)
        except ssl.SSLError as err:
            self.error("%s while retrieving page %s", err, page)
            return None
        except xmlrpclib.Fault as err:
            self.error("Failed call renderContent('%s','%s') : %d : %s",
                       space, page, err.faultCode, err.faultString)
            raise err

    @unsupported_in_api_version_1
    def convert_wiki_to_storage_format(self, markup):
        """
        Converts a wiki text to it's XML/HTML format. Useful if you prefer to
        generate pages using wiki syntax instead of XML.

        Still, remember that once you cannot retrieve the original wiki text,
        as confluence is not storing it anymore. \
        Due to this wiki syntax is usefull only for computer generated pages.

        Warning: this works only with Conflucence 4.0 or newer, on older
        versions it will raise an error.

        :param markup:
        :return:
        """
        return self._api.convertWikiToStorageFormat(self.token, markup)

    @unsupported_in_api_version_1
    def get_spaces(self):
        return self._api.getSpaces(self.token)

    @unsupported_in_api_version_1
    def get_pages(self, space):
        return self._api.getPages(self.token, space)

    unsupported_in_api_version_1 = staticmethod(unsupported_in_api_version_1)
