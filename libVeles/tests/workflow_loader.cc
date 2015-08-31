/*! @file workflow_loader.cc
 *  @brief WorkflowLoader tests.
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

#define ANTIDOTE
#include <veles/veles.h>
#include <gtest/gtest.h>
#include "tests/package_mixin.h"

namespace veles {

class All2AllTanh : public Unit {
 public:
  All2AllTanh(const std::shared_ptr<Engine>&)
     : Unit(nullptr), arrays_are_ok(false) {}

  virtual const std::string& Uuid() const noexcept override {
    return uuid_;
  }

  virtual void SetParameter(
      const std::string& name, const Property& value) override {
    if (name == "weights_transposed") {
      parameters_check[name] = value.is<bool>();
    } else if (name == "weights") {
      parameters_check[name] = value.is<PackagedNumpyArray>();
      auto array = value.get<PackagedNumpyArray>().get<float, 2, true>();
      arrays_are_ok = array.shape[0] == 100 && array.shape[1] == 784 &&
          array.transposed;
    } else if (name == "include_bias") {
      parameters_check[name] = value.is<bool>();
    } else if (name == "bias") {
      parameters_check[name] = value.is<PackagedNumpyArray>();
    } else if (name == "activation_mode") {
      parameters_check[name] = value.is<std::string>();
    }
  }

  virtual size_t OutputSize() const noexcept override {
    return 1024;
  }

  virtual void Execute() override {
  }

  std::unordered_map<std::string, bool> parameters_check;
  bool arrays_are_ok;

 private:
  static const std::string uuid_;
};

const std::string All2AllTanh::uuid_ = "b3a2bd5c-3c01-46ef-978a-fef22e008f31";

REGISTER_UNIT(All2AllTanh);

class All2AllSoftmax : public Unit {
 public:
  All2AllSoftmax(const std::shared_ptr<Engine>&) : Unit(nullptr) {}

  virtual const std::string& Uuid() const noexcept override {
    return uuid_;
  }

  virtual void SetParameter(const std::string&, const Property&) override {
  }

  virtual size_t OutputSize() const noexcept override {
    return 2048;
  }

  virtual void Execute() override {
  }

 private:
  static const std::string uuid_;
};

const std::string All2AllSoftmax::uuid_ = "420219fc-3e1a-45b1-87f8-aaa0c1540de4";

REGISTER_UNIT(All2AllSoftmax);

struct WorkflowLoaderTest : PackageMixin<WorkflowLoaderTest> {
};

TEST_F(WorkflowLoaderTest, FakeUnitsLoad) {
  auto w = WorkflowLoader().Load(
      path_to_files + "mnist.tar.gz", GetEngine());
  EXPECT_EQ(2, w.Size());
  auto all2all = reinterpret_cast<All2AllTanh*>(w.Head().get());
  EXPECT_TRUE(all2all->arrays_are_ok);
  int props = 0;
  for (auto& p : all2all->parameters_check) {
    if (p.second) {
      props++;
    }
  }
  EXPECT_EQ(5, props);
}

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
