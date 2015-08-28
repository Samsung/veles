/*! @file package_mixin.h
 *  @brief Mixin to discover the test packages.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>, Bulychev Egor <e.bulychev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013 Samsung R&D Institute Russia
 *
 *  @section License
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

#ifndef TESTS_PACKAGE_MIXIN_H_
#define TESTS_PACKAGE_MIXIN_H_

#include <veles/logger.h>

namespace veles {

template <class T>
struct PackageMixin :
    public virtual ::testing::Test,
    protected virtual DefaultLogger<T, Logger::COLOR_CYAN> {
  virtual void SetUp() override {
    char currentPath[FILENAME_MAX];
    assert(getcwd(currentPath, sizeof(currentPath)));
    std::string paths[] = { "/workflow_files/", "/tests/workflow_files/",
        "/../workflow_files/", "/../tests/workflow_files/" };
    INF("Current directory: %s\n", currentPath);
    for (auto& path : paths) {
      std::string tmp(currentPath);
      tmp += path;
      DBG("Probing %s...", tmp.c_str());
      if (access(tmp.c_str(), 0) != -1) {
        INF("Success: path is %s", tmp.c_str());
        path_to_files = tmp;
        return;
      }
    }
    assert(false && "Unable to locate the path with test packages");
  }

  std::string path_to_files;
};

}  // namespace veles

#endif  // TESTS_PACKAGE_MIXIN_H_
