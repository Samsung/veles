#!/bin/bash

find \( -name "*.cc" -o -name "*.h" \) -not -path "./tests/google/*" -not -path "./src/boost/*" -not -path "./inc/boost/*" -not -path "./eina/*" -not -path "./inc/veles/eina_*" -not -path "*/libarchive/*" -not -path "./zlib/*" -not -path "./yaml-cpp/*" -not -name "config.h" -print0 | xargs -0 ./cpplint.py --verbose=0 --filter=+,-build/include_alpha,-build/include_what_you_use,-whitespace/operators 2>&1 | ./cpplint_to_cppcheckxml.py 2> cpplint.xml
