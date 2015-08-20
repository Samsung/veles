/*! @file numpy_array_loader.cc
 *  @brief Numpy array loading tests.
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
#include <veles/logger.h>  // NOLINT(*)
#include "src/main_file_loader.h"
#include "tests/imemstream.h"


class MainFileLoaderTest : public ::testing::Test {
};

static const char contents[] = "{\n    \"checksum\": \"13088da21585f7a4cd7023083405e2094c53d746_2\",\n    \"units\": [\n        {\n            \"class\": {\n                \"name\": \"All2AllTanh\",\n                \"uuid\": \"b3a2bd5c-3c01-46ef-978a-fef22e008f31\"\n            },\n            \"data\": {\n                \"activation_mode\": \"ACTIVATION_TANH\",\n                \"bias\": \"@0000_100\",\n                \"include_bias\": true,\n                \"weights\": \"@0001_100x784\",\n                \"weights_transposed\": false\n            },\n            \"links\": [\n                1\n            ]\n        },\n        {\n            \"class\": {\n                \"name\": \"All2AllSoftmax\",\n                \"uuid\": \"420219fc-3e1a-45b1-87f8-aaa0c1540de4\"\n            },\n            \"data\": {\n                \"activation_mode\": \"ACTIVATION_LINEAR\",\n                \"bias\": \"@0002_10\",\n                \"include_bias\": true,\n                \"weights\": \"@0003_10x100\",\n                \"weights_transposed\": false\n            },\n            \"links\": []\n        }\n    ],\n    \"workflow\": \"MnistWorkflow\"\n}";

TEST(MainFileLoaderTest, Proof) {
  imemstream ms(contents, sizeof(contents));
  veles::MainFileLoader loader;
  auto wd = loader.Load(&ms);
  EXPECT_EQ("MnistWorkflow", wd.name());
  EXPECT_EQ("13088da21585f7a4cd7023083405e2094c53d746_2", wd.checksum());
  auto unit = wd.start();
  ASSERT_TRUE(static_cast<bool>(unit));
  EXPECT_EQ("All2AllTanh", unit->name());
  EXPECT_EQ("b3a2bd5c-3c01-46ef-978a-fef22e008f31", unit->uuid_str());
  EXPECT_EQ(1, unit->links().size());
  ASSERT_EQ(5, unit->PropertyNames().size());
  auto var1 = unit->get<std::string>("activation_mode");
  EXPECT_EQ("ACTIVATION_TANH", var1);
  auto var2 = unit->get<bool>("include_bias");
  EXPECT_TRUE(var2);
  var2 = unit->get<bool>("weights_transposed");
  EXPECT_FALSE(var2);
  auto var3 = unit->get<veles::NumpyArrayReference>("bias");
  EXPECT_EQ("0000_100.npy", var3.file_name());
  var3 = unit->get<veles::NumpyArrayReference>("weights");
  EXPECT_EQ("0001_100x784.npy", var3.file_name());
  unit = unit->links()[0];
  EXPECT_EQ("All2AllSoftmax", unit->name());
  EXPECT_EQ("420219fc-3e1a-45b1-87f8-aaa0c1540de4", unit->uuid_str());
  EXPECT_EQ(0, unit->links().size());
  ASSERT_EQ(5, unit->PropertyNames().size());
}


#include "tests/google/src/gtest_main.cc"

