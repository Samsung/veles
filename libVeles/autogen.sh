#!/bin/sh

#  This file is a part of SEAPT, Samsung Extended Autotools Project Template

#  Copyright 2012,2013 Samsung R&D Institute Russia
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met: 
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer. 
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

isubuntu="$(uname -v|grep Ubuntu)"

check_prog() {
    printf "Checking for $1... "
    if [ -z "$($1 --version 2>/dev/null)" ]; then
        echo no
        if [ -n "$isubuntu" ]; then
            sudo apt-get -y install $2
            printf "Checking for $1... "
            if [ -z "$($1 --version 2>/dev/null)" ]; then
                echo no
                echo "Error: $1 was not found, aborting"
                exit 1
            fi
        else
            echo "Error: $1 was not found, aborting"
            exit 1
        fi
    fi
    echo yes
}

check_prog aclocal automake
check_prog autoheader autoconf
check_prog autoconf autoconf
check_prog libtoolize libtool
check_prog automake automake

rm -rf autom4te.cache m4
rm -f aclocal.m4 ltmain.sh config.log config.status configure libtool stamp-h1 config.h config.h.in
find -name Makefile.in -exec rm {} \;

echo "Running aclocal..." ; aclocal $ACLOCAL_FLAGS || exit 1
echo "Running autoheader..." ; autoheader || exit 1
echo "Running autoconf..." ; autoconf || exit 1
echo "Running libtoolize..." ; (libtoolize --copy --automake || glibtoolize --automake) || exit 1
echo "Running automake..." ; (automake --add-missing --copy --foreign ) || exit 1

W=0

rm -f config.cache-env.tmp
echo "OLD_PARM=\"$@\"" >> config.cache-env.tmp
echo "OLD_CFLAGS=\"$CFLAGS\"" >> config.cache-env.tmp
echo "OLD_CXXFLAGS=\"$CXXFLAGS\"" >> config.cache-env.tmp
echo "OLD_CPPFLAGS=\"$CPPFLAGS\"" >> config.cache-env.tmp
echo "OLD_PATH=\"$PATH\"" >> config.cache-env.tmp
echo "OLD_PKG_CONFIG_PATH=\"$PKG_CONFIG_PATH\"" >> config.cache-env.tmp
echo "OLD_LDFLAGS=\"$LDFLAGS\"" >> config.cache-env.tmp

cmp -s config.cache-env.tmp config.cache-env >> /dev/null
if [ $? -ne 0 ]; then
	W=1;
fi

if [ $W -ne 0 ]; then
	echo "Cleaning configure cache...";
	rm -f config.cache config.cache-env
	mv config.cache-env.tmp config.cache-env
else
	rm -f config.cache-env.tmp
fi

if [ -n "$1" ]; then
	path=$(pwd)
	mkdir -p "$1"
	cd "$1"
	shift
	$path/configure $@
	cd $path
fi
