##  This file is a part of SEAPT, Samsung Extended Autotools Project Template

##  Copyright 2012,2013 Samsung R&D Institute Russia
##  All rights reserved.
##
##  Redistribution and use in source and binary forms, with or without
##  modification, are permitted provided that the following conditions are met: 
##
##  1. Redistributions of source code must retain the above copyright notice, this
##     list of conditions and the following disclaimer. 
##  2. Redistributions in binary form must reproduce the above copyright notice,
##     this list of conditions and the following disclaimer in the documentation
##     and/or other materials provided with the distribution.
##
##  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
##  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
##  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
##  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
##  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
##  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
##  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
##  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
##  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
##  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if HAVE_DOXYGEN
directory = $(top_builddir)/docs/html 

$(directory)index.html: doxyfile.stamp

doxyfile.stamp: Doxyfile
	$(AM_V_GEN)$(DOXYGEN) Doxyfile
	$(AM_V_at)echo Timestamp > doxyfile.stamp

CLEANFILES = doxyfile.stamp

all-local: $(directory)index.html doxyfile.stamp

install-data-local: doxyfile.stamp
	$(AM_V_GEN)$(MKDIR_P) $(DESTDIR)$(htmldir)/search && \
	for file in $$(find $(top_builddir)/docs/html -maxdepth 1 -type f); do \
		$(INSTALL_DATA) $$file $(DESTDIR)$(htmldir); \
	done && \
	for file in $$(find $(top_builddir)/docs/html/search -maxdepth 1 -type f); do \
		$(INSTALL_DATA) $$file $(DESTDIR)$(htmldir)/search; \
	done

uninstall-local:
	$(AM_V_at)rm -rf $(DESTDIR)$(htmldir)

clean-local:
	$(AM_V_at)rm -rf $(directory)
	
endif
