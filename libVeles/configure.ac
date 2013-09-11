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

AC_INIT(Veles, 1.0)
m4_include([common.ac])
AC_SUBST(INTERFACE_VERSION, [0])
AC_SUBST(REVISION_NUMBER, [$(git rev-list HEAD --count)])
AC_SUBST(AGE_NUMBER, [0])

# Check whether to use the built-in libarchive
AC_ARG_WITH([built-in-libarchive],
    AS_HELP_STRING([--with-built-in-libarchive], [use bundled libarchive]), [
    LIBARCHIVE_LIBS="\$(top_builddir)/libarchive/libarchive.la"
    LIBARCHIVE_CFLAGS="-I\$(top_srcdir)/libarchive/libarchive"
    LIBARCHIVE_DIR=libarchive
    AC_CONFIG_SUBDIRS([libarchive])
    ZLIB_DIR=zlib
    STATIC_ZLIB_LIBS="\$(top_builddir)/zlib/libz.a"
    echo "$AM_CFLAGS $CFLAGS $AM_CPPFLAGS $CPPFLAGS" > zlib.cflags
    sed -i -e 's/-I$@<:@^ @:>@*//g' -e 's/--coverage//g' zlib.cflags
    AC_CONFIG_COMMANDS([zlib],
                       [ln -sf $ac_top_srcdir/zlib zlib && \
                       cd zlib && \
                       CC=$CC CFLAGS="$(cat ../zlib.cflags) -fPIC -Wno-implicit-function-declaration" \
                       ./configure --static --const && \
                       cd ..])
], [
    # Check for installed libarchive
    PKG_CHECK_MODULES([LIBARCHIVE], [libarchive >= 3.0])
])
AC_SUBST([LIBARCHIVE_DIR])
AC_SUBST([ZLIB_DIR])
AC_SUBST([STATIC_ZLIB_LIBS])

AS_IF([test "x$enable_static" != "xno" -a "x$enable_shared" != "xno" ], [
    AC_MSG_ERROR([Building both shared and static libraries is not supported (use --disable-shared or --disable-static)])
])

# Check whether to use the built-in yaml-cpp
AC_ARG_WITH([built-in-yaml-cpp],
    AS_HELP_STRING([--with-built-in-yaml-cpp], [use bundled yaml-cpp]), [
    STATIC_YAMLCPP_LIBS="\$(top_builddir)/yaml-cpp/libyaml-cpp.a"
    YAMLCPP_CFLAGS="-I\$(top_srcdir)/yaml-cpp/include"
    YAMLCPP_DIR=yaml-cpp
    echo "$AM_CXXFLAGS $CXXFLAGS $AM_CPPFLAGS $CPPFLAGS" > yaml-cpp.cxxflags
    sed -i -e 's/-I$@<:@^ @:>@*//g' -e 's/--coverage//g' yaml-cpp.cxxflags
    mkdir -p yaml-cpp    
    AS_IF([test "$host" = "$build" ], [
        AC_CONFIG_COMMANDS([cmake], [
        cd yaml-cpp
        CXXFLAGS="$(cat ../yaml-cpp.cxxflags) \
        	-Wno-old-style-cast -Wno-deprecated-declarations -fPIC" \
        cmake ../$ac_top_srcdir/yaml-cpp \
            -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF \
            -DBUILD_SHARED_LIBS=OFF
        cd ..])
    ], [
        echo "set(CMAKE_SYSTEM_NAME Linux)" > $host.cmake
        echo "set(CMAKE_SYSTEM_VERSION 1)" >> $host.cmake
        echo "set(CMAKE_C_COMPILER $CC)" >> $host.cmake
        echo "set(CMAKE_CXX_COMPILER $CXX)" >> $host.cmake
        AC_CONFIG_COMMANDS(cmake_cross, [
        cd yaml-cpp
        CXXFLAGS="$(cat ../yaml-cpp.cxxflags) \
        	-Wno-old-style-cast -Wno-deprecated-declarations -fPIC" \
        cmake ../$ac_top_srcdir/yaml-cpp \
            -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF \
            -DBUILD_SHARED_LIBS=OFF \
            -DCMAKE_TOOLCHAIN_FILE=../$host.cmake
        cd ..])
    ])
], [
    # Check for installed libarchive
    PKG_CHECK_MODULES([YAMLCPP], [yaml-cpp >= 0.5])
])
AC_SUBST([YAMLCPP_DIR])
AC_SUBST([STATIC_YAMLCPP_LIBS])

AM_COND_IF([TESTS], [
    AC_CONFIG_LINKS([tests/workflow_files/remove_folder_testing.tar.gz:tests/workflow_files/remove_folder_testing.tar.gz
                     tests/workflow_files/test_archive.tar.gz:tests/workflow_files/test_archive.tar.gz
                     tests/workflow_files/default.yaml:tests/workflow_files/default.yaml
                     tests/workflow_files/neural_network.tar.gz:tests/workflow_files/neural_network.tar.gz
                     tests/workflow_files/workflow.tar.gz:tests/workflow_files/workflow.tar.gz])    
])

AC_OUTPUT
