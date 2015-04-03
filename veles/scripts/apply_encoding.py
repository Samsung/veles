#!/usr/bin/env python3
# encoding: utf-8
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 3, 2015

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import os
import sys

from veles.paths import __root__


def main():
    path = __root__
    for rt_path, _tmp, files in os.walk(path, followlinks=True):
        for file in files:
            full_path = os.path.join(rt_path, file)
            if full_path.endswith(".py"):
                encoding = False
                shibang = False
                with open(full_path, "r") as fin:
                    first_line_is = True
                    all_lines = []
                    for line in fin:
                        if first_line_is:
                            first_line = line
                            first_line_is = False
                        else:
                            all_lines.append(line)
                        if first_line and first_line.find("#!/usr/") >= 0:
                            shibang = True
                        if line.find("utf-8") > 0:
                            encoding = True
                            break
                if not encoding:
                    if not shibang:
                        with open(full_path, "r") as fin:
                            data = fin.read()
                        with open(full_path, "w") as fout:
                            fout.write("# -*- coding: utf-8 -*-\n" + data)
                    else:
                        with open(full_path, "w") as fout:
                            fout.write("#!/usr/bin/env python3\n# -*-codi" +
                                       "ng: utf-8 -*-\n")
                        with open(full_path, "a") as fout:
                            for line in all_lines:
                                fout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
