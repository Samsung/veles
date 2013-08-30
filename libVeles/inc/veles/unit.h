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
#include <memory>
#include <veles/poison.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace Veles {

/** @brief VELES neural network unit */
class Unit {
 public:
  virtual ~Unit() = default;
  /** @brief Name of the unit
   */
  virtual std::string Name() const noexcept = 0;
  /** @brief Sets or modifies a parameter
   *  @param name Parameter name
   *  @param value Pointer to parameter data
   */
  virtual void SetParameter(const std::string& name,
                            std::shared_ptr<void> value) = 0;
  /** @brief Executes this unit on input data
   *  @param in Input vector
   *  @param out Output vector
   */
  virtual void Execute(const float* in, float* out) const = 0;
  /* @brief Number of unit inputs
   */
  virtual size_t InputCount() const noexcept = 0;
  /* @brief Number of unit outputs
   */
  virtual size_t OutputCount() const noexcept = 0;
};

}  // namespace Veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_UNIT_H_
