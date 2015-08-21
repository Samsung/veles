/*! @file workflow_loader.cc
 *  @brief Tests for class WorkflowLoder.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
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

#include <cassert>
#include <gtest/gtest.h>
#include <veles/workflow_loader.h>
#include "src/workflow_archive.h"


namespace veles {

struct WorkflowLoaderTest :
    public ::testing::Test,
    protected DefaultLogger<WorkflowLoader, Logger::COLOR_CYAN> {
  std::shared_ptr<WorkflowArchive> ExtractArchive(const std::string& file_name) {
    return WorkflowLoader().ExtractArchive(file_name);
  }

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

TEST_F(WorkflowLoaderTest, MnistWorkflow) {
  auto wa = ExtractArchive(path_to_files + "mnist.tar.gz");
  ASSERT_TRUE(static_cast<bool>(wa));
  auto& wdef = wa->workflow_definition();
  ASSERT_EQ("MnistWorkflow", wdef.name());
}

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
