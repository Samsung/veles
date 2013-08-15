/*! @file unit_registry.cc
 *  @brief Tests for Veles::UnitRegistry and friends.
 *  @author markhor
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */



#include <gtest/gtest.h>
#include "inc/veles/unit_registry.h"

namespace Veles {

class DummyUnit : public Unit {
 public:
  virtual std::string Name() const noexcept override {
    return "Dummy";
  }

  virtual void SetParameter(const std::string&,
                            std::shared_ptr<void>) override {
  }

  virtual void Execute(float*, float*) const override {
  }

  virtual size_t InputCount() const noexcept {
    return 0;
  }

  virtual size_t OutputCount() const noexcept {
    return 0;
  }
};
  REGISTER_UNIT(DummyUnit);
}


TEST(UnitRegistry, DummyCreate) {
  auto dummy = Veles::UnitFactory::Instance()["Dummy"]();
  ASSERT_STREQ("Dummy", dummy->Name().c_str());
}

#include "tests/google/src/gtest_main.cc"
