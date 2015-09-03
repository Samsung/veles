/*! @file workflow.cc
 *  @brief Workflow class tests.
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

#define ANTIDOTE
#include <veles/engine.h>
#include <veles/workflow.h>
#include <veles/unit.h>
#include <gtest/gtest.h>

namespace veles {

class DummyUnit : public Unit {
 public:
  DummyUnit(const std::shared_ptr<Engine>& e,
            int* in, int* ex) : Unit(e), initialized(in), executed(ex) {}

  virtual const std::string& Uuid() const noexcept override {
    return uuid_;
  }

  virtual void SetParameter(const std::string&, const Property&) override {
  }

  virtual size_t OutputSize() const noexcept override {
    return 100;
  }

  virtual void Initialize() override {
    Unit::Initialize();
    *initialized = true;
  }

  virtual void Execute() override {
    *executed = true;
  }

  int* initialized;
  int* executed;

 private:
  static const std::string uuid_;
};

const std::string DummyUnit::uuid_ = "";

class DummyEngine : public Engine {
 public:
  virtual void Schedule(const Callable&) override {
  }
};

TEST(Workflow, Initialize) {
  auto engine = std::make_shared<DummyEngine>();
  std::vector<int> initialized { 0, 0, 0, 0 };
  std::vector<int> executed { 0, 0, 0, 0 };
  auto head = std::make_shared<DummyUnit>(
      engine, initialized.data(), executed.data());
  auto second = std::make_shared<DummyUnit>(
      engine, initialized.data() + 1, executed.data() + 1);
  second->LinkFrom(head);
  auto third = std::make_shared<DummyUnit>(
      engine, initialized.data() + 2, executed.data() + 2);
  third->LinkFrom(head);
  auto fourth = std::make_shared<DummyUnit>(
      engine, initialized.data() + 3, executed.data() + 3);
  fourth->LinkFrom(second);
  fourth->LinkFrom(third);
  DummyUnit* units[] { head.get(), second.get(), third.get(), fourth.get() };
  Workflow workflow("TestWorkflow", "checksum", head, engine);
  for (auto& unit : units) {
    EXPECT_FALSE(*unit->initialized);
    EXPECT_FALSE(*unit->executed);
    EXPECT_EQ(&workflow, unit->workflow());
  }
  alignas(64) float data[10];
  workflow.Initialize(data);
  EXPECT_EQ(data, workflow.input());
  for (auto& unit : units) {
    EXPECT_TRUE(*unit->initialized);
  }
  workflow.Run();
}

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
