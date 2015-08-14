/*! @file unit.h
 *  @brief VELES neural network unit
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
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

#ifndef INC_VELES_UNIT_H_
#define INC_VELES_UNIT_H_

#include <string>
#include <memory>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

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
                            std::shared_ptr<const void> value) = 0;
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

}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_UNIT_H_
