/*! @file unit_registry.cc
 *  @brief Tests for veles::UnitRegistry and friends.
 *  @author markhor
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

#include <gtest/gtest.h>
#include "inc/veles/unit_factory.h"

namespace veles {

class DummyUnit : public virtual Unit,
                  public virtual DefaultLogger<DummyUnit, Logger::COLOR_ORANGE> {
 public:
  DummyUnit() : Unit(nullptr) {}

  virtual const std::string& Uuid() const noexcept override {
    return uuid_;
  }

  virtual void SetParameter(const std::string&, const Property&) override {
  }

  virtual size_t OutputSize() const override {
    return 0;
  }

  virtual void Execute() override {
  }

 private:
  static const std::string uuid_;
};

const std::string DummyUnit::uuid_ = "abcd";

REGISTER_UNIT(DummyUnit);

}  // namespace veles


TEST(UnitRegistry, DummyCreate) {
  auto dummy = veles::UnitFactory::Instance()["abcd"]();
  ASSERT_STREQ("abcd", dummy->Uuid().c_str());
}

#include "tests/google/src/gtest_main.cc"
