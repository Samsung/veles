/*! @file workflow_archive.cc
 *  @brief Tests for WorkflowArchive class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2015 Samsung R&D Institute Russia
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
#include "src/workflow_archive.h"
#include "src/numpy_array_loader.h"


namespace veles {

namespace internal {

struct WorkflowArchiveTest :
    public ::testing::Test,
    protected DefaultLogger<WorkflowArchive, Logger::COLOR_CYAN> {
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

TEST_F(WorkflowArchiveTest, MnistWorkflow) {
  auto wa = WorkflowArchive::Load(path_to_files + "mnist.tar.gz");
  ASSERT_TRUE(static_cast<bool>(wa));
  auto& wdef = wa->workflow_definition();
  ASSERT_EQ("MnistWorkflow", wdef.name());
  ASSERT_EQ("All2AllTanh", wdef.start()->name());
  ASSERT_EQ(1, wdef.start()->links().size());
  ASSERT_EQ("All2AllSoftmax", wdef.start()->links()[0]->name());
  ASSERT_EQ(4, wa->files_.size());
}

TEST_F(WorkflowArchiveTest, GetStream) {
  auto wa = WorkflowArchive::Load(path_to_files + "mnist.zip");
  auto stream = wa->GetStream("0001_100x784.npy");
  NumpyArrayLoader nal;
  auto array = nal.Load<float, 2, true>(stream.get());
  ASSERT_EQ(2, array.shape.size());
  EXPECT_EQ(100, array.shape[0]);
  EXPECT_EQ(784, array.shape[1]);
  EXPECT_EQ(100 * 784, array.data.size());
}

}  // namespace internal

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
