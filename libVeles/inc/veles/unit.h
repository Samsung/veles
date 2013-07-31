/*! @file unit.h
 *  @brief VELES neural network unit
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_VELES_UNIT_H_
#define INC_VELES_UNIT_H_

#include <string>

namespace Veles {

/** @brief VELES neural network unit */
class Unit {
 public:
  virtual ~Unit() noexcept {
  }

  virtual std::string Name() const noexcept = 0;

  virtual void Load(const std::string& data) = 0;
  virtual void Execute(float* in, float* out) const = 0;
  virtual size_t inputs() const noexcept = 0;
  virtual size_t outputs() const noexcept = 0;
};

}  // namespace Veles

#endif  // INC_VELES_UNIT_H_
