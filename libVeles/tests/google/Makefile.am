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

#######################################
# We are building a static library lib_gtest.a with libtool
noinst_LTLIBRARIES = lib_gtest.la

#######################################
# Build information for lib_gtest.la

# Sources for lib_gtest.a
lib_gtest_la_SOURCES = src/gtest-all.cc

# Compiler options. Here we are adding the include directory
# to be searched for headers included in the source code.
lib_gtest_la_CXXFLAGS = -I$(top_srcdir)/tests/google \
	-Wno-missing-field-initializers -Wno-old-style-cast

.PHONY: tests

tests: all
	@echo >/dev/null
