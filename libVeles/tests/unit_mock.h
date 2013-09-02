/*! @file unit_mock.h
 *  @brief Mock unit that is used to test VELES Workflow independently of Znicz.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef TESTS_UNIT_MOCK_H_
#define TESTS_UNIT_MOCK_H_

#include <gtest/gtest.h>
#include "inc/veles/unit.h"

namespace Simd {

float* mallocf(size_t length) {
  void *ptr;
  return posix_memalign(&ptr, 64, length * sizeof(float)) == 0
      ? static_cast<float*>(ptr) : nullptr;
}

};

/** @brief Dummy unit which passes doubled looped inputs to outputs.
 */
class UnitMock : public Veles::Unit {
 public:
  static const std::string kName;

  UnitMock(size_t inputs = 0, size_t outputs = 0) {
    Initialize(inputs, outputs);
  }
  void Initialize(size_t inputs = 0, size_t outputs = 0) {
    inputs_ = inputs;
    outputs_ = outputs;
  }
  virtual std::string Name() const noexcept override {
      return kName;
    }
  virtual void SetParameter(const std::string& /* name */,
                            std::shared_ptr<void> /* value */) override {
  }
  virtual void Execute(const float* in, float* out) const override {
    for (size_t i = 0; i < OutputCount(); ++i) {
      out[i] = in[i % InputCount()] * 2;
    }
  }
  virtual size_t InputCount() const noexcept override {
    return inputs_;
  }
  virtual size_t OutputCount() const noexcept override {
    return outputs_;
  }

 private:
  size_t inputs_;
  size_t outputs_;
};

class UnitMockTest : public ::testing::TestWithParam<
                              std::tuple<size_t, size_t>> {
 protected:
  virtual void SetUp() override;

  UnitMock unit_;
};

const std::string UnitMock::kName = "UnitMock";

#endif  // TESTS_UNIT_MOCK_H_
