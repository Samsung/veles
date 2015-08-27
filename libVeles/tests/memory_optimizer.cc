/*! @file memory_optimizer.cc
 *  @brief Memory scheduler tests.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2015 © Samsung R&D Institute Russia
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


#include <gtest/gtest.h>
#include "src/memory_optimizer.h"
#include <sstream>

namespace veles {

namespace internal {

class MemoryOptimizerTest : public ::testing::Test {
};

TEST_F(MemoryOptimizerTest, Linear) {
  auto opt = MemoryOptimizer();
  std::vector<MemoryNode> nodes(5);
  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    nodes[i].time_start = i;
    nodes[i].time_finish = i + 2;
    nodes[i].value = 1;
  }
  size_t max = opt.Optimize(&nodes);
  ASSERT_EQ(2, max) << max;
  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    ASSERT_EQ(i % 2, nodes[i].position) << i;
  }
}

TEST_F(MemoryOptimizerTest, Twisted) {
  auto opt = MemoryOptimizer();
  std::vector<MemoryNode> nodes(7);
  nodes[0].time_start = 0;
  nodes[0].time_finish = 5;
  nodes[0].value = 2;

  nodes[1].time_start = 1;
  nodes[1].time_finish = 6;
  nodes[1].value = 1;
  nodes[2].time_start = 1;
  nodes[2].time_finish = 4;
  nodes[2].value = 1;
  nodes[3].time_start = 1;
  nodes[3].time_finish = 4;
  nodes[3].value = 1;

  nodes[4].time_start = 1;
  nodes[4].time_finish = 7;
  nodes[4].value = 1;

  nodes[5].time_start = 4;
  nodes[5].time_finish = 7;
  nodes[5].value = 2;

  nodes[6].time_start = 6;
  nodes[6].time_finish = 7;
  nodes[6].value = 3;
  size_t max = opt.Optimize(&nodes);
  ASSERT_EQ(6, max) << max;

  std::stringstream str;
  opt.Print(nodes, &str);
  printf("%s", str.str().c_str());
  ASSERT_TRUE(str.str().size());
}

}  // namespace internal

}  // namespace veles

#include "tests/google/src/gtest_main.cc"

