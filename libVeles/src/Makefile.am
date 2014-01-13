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

include $(top_srcdir)/Makefile.common
include Sources.make

lib_LTLIBRARIES = libVeles.la

#######################################
# Build information for libVeles.la

# Source files
libVeles_la_SOURCES = $(SOURCES)

# Compiler options
libVeles_la_CPPFLAGS = $(AM_CPPFLAGS) @LIBARCHIVE_CFLAGS@ @YAMLCPP_CFLAGS@

# Used libraries
libVeles_la_LIBADD = @LIBARCHIVE_LIBS@ @YAMLCPP_LIBS@ @EINA_LIBS@

libVeles_la_LDFLAGS = $(AM_LDFLAGS) \
	-version-info $(INTERFACE_VERSION):$(REVISION_NUMBER):$(AGE_NUMBER) \
	-Wl,--whole-archive,$(STATIC_ZLIB_LIBS),--no-whole-archive \
	-Wl,--whole-archive,$(STATIC_YAMLCPP_LIBS),--no-whole-archive

PARALLEL_SUBDIRS =
