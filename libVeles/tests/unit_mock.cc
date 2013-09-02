/*! @file unit_mock.cc
 *  @brief Unit Mock tests.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "tests/unit_mock.h"
#include <cstdlib>
#include "inc/veles/make_unique.h"


void UnitMockTest::SetUp() {
  size_t inputs = 0;
  size_t outputs = 0;
  std::tie(inputs, outputs) = GetParam();
  unit_.Initialize(inputs, outputs);
}

TEST_P(UnitMockTest, Execute) {
  size_t inputs = 0;
  size_t outputs = 0;
  std::tie(inputs, outputs) = GetParam();
  ASSERT_NE(static_cast<size_t>(0), inputs * outputs);
  auto input  = std::uniquify(Simd::mallocf(inputs), std::free);
  auto output = std::uniquify(Simd::mallocf(outputs), std::free);
  for (size_t i = 0; i < inputs; ++i) {
    input.get()[i] = i;
  }
  unit_.Execute(input.get(), output.get());
  float expected_multiply = 2;
  for (size_t i = 0; i < outputs; ++i) {
    ASSERT_NEAR((i % inputs) * expected_multiply, output.get()[i], 0.01)
        << "i = " << i << std::endl;
  }
}

INSTANTIATE_TEST_CASE_P(
    UnitMockTests, UnitMockTest,
    ::testing::Combine(
        ::testing::Values(5, 15),
        ::testing::Values(7, 24)));

#include "tests/google/src/gtest_main.cc"


