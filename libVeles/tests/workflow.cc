/*! @file workflow.cc
 *  @brief VELES workflow tests
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include "inc/veles/workflow.h"
#include "inc/veles/make_unique.h"
#include "tests/workflow.h"

namespace {

float* mallocf(size_t length) {
  void *ptr;
  return posix_memalign(&ptr, 64, length * sizeof(float)) == 0
      ? static_cast<float*>(ptr) : nullptr;
}

}

TEST(Workflow, Construct) {
  Veles::Workflow workflow;
  size_t kExpectedCount = 0;
  EXPECT_EQ(kExpectedCount, workflow.UnitCount());
}

TEST(Workflow, Add) {
  Veles::Workflow workflow;
  const size_t kCount = 100;
  const size_t kInputs = 3;
  const size_t kOutputs = 4;
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(i, workflow.UnitCount());
    workflow.AddUnit(std::make_shared<UnitMock>(kInputs, kOutputs));
  }
}

TEST(Workflow, Execute) {
  Veles::Workflow workflow;
  const size_t kCount = 4;
  const size_t kInputs = 10;
  for (size_t i = 0; i < kCount; ++i) {
    workflow.AddUnit(std::make_shared<UnitMock>(kInputs, kInputs));
  }
  auto input  = std::uniquify(mallocf(kInputs), std::free);
  auto output = std::uniquify(mallocf(kInputs), std::free);
  for (size_t i = 0; i < kInputs; ++i) {
    input.get()[i] = i;
  }
  workflow.Execute(input.get(), input.get() + kInputs, output.get());
  float expected_multiply = std::pow(2, kCount);
  for (size_t i = 0; i < kInputs; ++i) {
    ASSERT_NEAR(i * expected_multiply, output.get()[i], 0.01);
  }
}

#include "tests/google/src/gtest_main.cc"
