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

#include "tests/workflow.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "inc/veles/make_unique.h"
#include "tests/unit_mock.h"

const size_t WorkflowTest::kCount = 3;

TEST(Workflow, Construct) {
  veles::Workflow workflow;
  size_t kExpectedCount = 0;
  EXPECT_EQ(kExpectedCount, workflow.Size());
}

TEST(Workflow, Add) {
  veles::Workflow workflow;
  const size_t kCount = 100;
  const size_t kInputs = 3;
  const size_t kOutputs = 4;
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(i, workflow.Size());
    bool isEven = i % 2 == 0;
    workflow.Add(
        std::make_shared<UnitMock>(
            isEven ? kInputs : kOutputs,
            isEven ? kOutputs : kInputs));
  }
  for (size_t i = 0; i < kCount; ++i) {
    bool isEven = i % 2 == 0;
    ASSERT_EQ(UnitMock::kName, workflow.Get(i)->Name());
    ASSERT_EQ(isEven ? kInputs : kOutputs, workflow.Get(i)->InputCount());
    ASSERT_EQ(isEven ? kOutputs : kInputs, workflow.Get(i)->OutputCount());
  }
  ASSERT_THROW(workflow.Get(kCount), std::out_of_range);
  ASSERT_THROW(workflow.Get(kCount * 2), std::out_of_range);
}

TEST(Workflow, Parameters) {
  veles::Workflow workflow;
  std::string param1_name = "abc";
  std::string param2_name = "def";
  std::string value1_str = "one";
  std::string value2_str = "two";
  std::shared_ptr<const std::string> value1(new std::string(value1_str));
  std::shared_ptr<const std::string> value2(new std::string(value2_str));
  std::shared_ptr<const std::vector<int>> value3(
      new std::vector<int>{1, 42, 99});
  // initial value

  EXPECT_EQ(nullptr, workflow.GetParameter(param1_name));

  // setting string parameter
  workflow.SetParameter(param1_name, value2);
  EXPECT_EQ(value2_str, *std::static_pointer_cast<const std::string>(
      workflow.GetParameter(param1_name)));
  // resetting parameter

  workflow.SetParameter(param1_name, value1);
  EXPECT_EQ(value1_str, *std::static_pointer_cast<const std::string>(
      workflow.GetParameter(param1_name)));

  // initial value of other parameter
  EXPECT_EQ(nullptr, workflow.GetParameter(param2_name));

  // setting vector parameter
  workflow.SetParameter(param2_name, value3);
  EXPECT_EQ(42, (*std::static_pointer_cast<const std::vector<int>>(
      workflow.GetParameter(param2_name)))[1]);
}

void WorkflowTest::SetUp() {
  size_t sizes[kCount + 1];
  std::tie(sizes[0], sizes[1], sizes[2], sizes[3]) = GetParam();
  for (size_t i = 0; i < kCount; ++i) {
    workflow_.Add(
        std::make_shared<UnitMock>(sizes[i], sizes[i + 1]));
  }
}

void WorkflowTest::GetExpected(std::vector<size_t>* out) {
  size_t sizes[kCount + 1];
  std::tie(sizes[0], sizes[1], sizes[2], sizes[3]) = GetParam();
  float expected_multiply = std::pow(2, kCount);
  for (size_t i = 0; i < sizes[0]; ++i) {
    out->push_back((i + 1) * expected_multiply);
  }
  for (size_t i = 1; i <= kCount; ++i) {
    size_t last = sizes[i - 1];
    ASSERT_NE(static_cast<size_t>(0), last);
    size_t curr = sizes[i];
    if (curr > last) {
      for (size_t j = 0; j < curr - last; ++j) {
        out->push_back((*out)[j % last]);
      }
    } else if (curr < last) {
      out->resize(curr);
    }
  }
}

TEST_P(WorkflowTest, Construct) {
  size_t inputs = 0;
  size_t outputs = 0;
  std::tie(inputs, std::ignore, std::ignore, outputs) = GetParam();
  EXPECT_EQ(kCount, workflow_.Size());
  EXPECT_EQ(inputs, workflow_.InputCount());
  EXPECT_EQ(outputs, workflow_.OutputCount());
}

TEST_P(WorkflowTest, Execute) {
  size_t sizes[kCount + 1];
  std::tie(sizes[0], sizes[1], sizes[2], sizes[3]) = GetParam();
  size_t inputs = sizes[0];
  size_t outputs = sizes[kCount];
  std::tie(inputs, std::ignore, std::ignore, outputs) = GetParam();
  auto input  = std::uniquify(Simd::mallocf(inputs), std::free);
  auto output = std::uniquify(Simd::mallocf(outputs), std::free);
  for (size_t i = 0; i < inputs; ++i) {
    input.get()[i] = i + 1;
  }
  workflow_.Execute(input.get(), input.get() + inputs, output.get());
  std::vector<size_t> expected;
  GetExpected(&expected);
  for (size_t i = 0; i < outputs; ++i) {
    ASSERT_NEAR(expected[i], output.get()[i], 0.01)
        << "i = " << i << std::endl;
  }
}

INSTANTIATE_TEST_CASE_P(
    WorkflowTests, WorkflowTest,
    ::testing::Combine(
        ::testing::Values(1, 7, 100),
        ::testing::Values(3, 14),
        ::testing::Values(5, 10),
        ::testing::Values(1, 9, 100)));

#include "tests/google/src/gtest_main.cc"
