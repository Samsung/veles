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

if PARALLEL_BUILD

$(PARALLEL_SUBDIRS)::
	$(MAKE) -C $@ $(MAKECMDGOALS)

all-local:: $(PARALLEL_SUBDIRS)

clean-local:: $(PARALLEL_SUBDIRS)
	rm -f *.xml

SUBDIRS = $(DEPENDENCY_SUBDIRS)

else

SUBDIRS = $(DEPENDENCY_SUBDIRS) $(PARALLEL_SUBDIRS)

clean-local::
	rm -f *.xml
	
endif

AM_DEFAULT_SOURCE_EXT = .cc

AM_CPPFLAGS = -I$(top_srcdir)/tests/google
AM_LDFLAGS = $(top_builddir)/src/libVeles.la \
       $(top_builddir)/tests/google/lib_gtest.la \
       -pthread

noinst_PROGRAMS = $(TESTS)

.PHONY: tests

REALLOG=$(top_builddir)/$(TESTLOG)

DEFAULT_TIMEOUT=10

tests:	
	@for dir in $(PARALLEL_SUBDIRS); do \
		cd $$dir; $(MAKE) --no-print-directory tests; cd ..; \
	done
	@echo [~~~~~~~~~~] >>$(REALLOG)
	@echo Running tests in $(srcdir) >>$(REALLOG)
	@echo [~~~~~~~~~~] >>$(REALLOG)
	@for et in $(TESTS); do \
	skip=0; \
	for nt in $(not_tests); do \
		if [ "$$et" == "$$nt" ]; then \
			skip=1; \
			break; \
		fi; \
	done; \
	if [ "$$skip" == "1" ]; then \
		continue; \
	fi; \
	if [ -z "$(TIMEOUT)" ]; then \
	    timeout_value=$(DEFAULT_TIMEOUT); \
	else \
		timeout_value=$(TIMEOUT); \
	fi; \
	timeout $$timeout_value /usr/bin/time -f "\"$$et\" peak memory usage: %M Kb" ./$$et --gtest_output="xml:$$et.xml" &>>$(REALLOG); \
	if [ "$$?" -eq "0" ]; then \
		echo -e "\033[01;32m[DONE]\033[00m $$et"; \
	else \
		echo -e "\033[01;31m[FAIL]\033[00m $$et"; \
		echo "[FAILED]" >>$(REALLOG); \
	fi; \
	done

