/*! @file workflow.h
 *  @brief New file description.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef TESTS_WORKFLOW_H_
#define TESTS_WORKFLOW_H_

#include "inc/veles/unit.h"

/** @brief Dummy unit which passes doubled looped inputs to outputs.
 */
class UnitMock : public Veles::Unit {
 public:
  UnitMock(size_t inputs, size_t outputs) : inputs_(inputs), outputs_(outputs) {
  }
  virtual std::string Name() const noexcept override {
      return "UnitMock";
    }
  virtual void SetParameter(const std::string& /* name */,
                            std::shared_ptr<void> /* value */) override {
  }
  virtual void Execute(const float* in, float* out) const override {
    for(size_t i = 0; i < OutputCount(); ++i) {
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


#endif  // TESTS_WORKFLOW_H_
