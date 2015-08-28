/*! @file unit.h
 *  @brief Unit class declaration.
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

#include <functional>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <variant/variant.hpp>
#include <veles/logger.h>
#include <veles/packaged_numpy_array.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

class Engine;

class UnexpectedRunException : public std::exception {
 public:
  UnexpectedRunException(const std::string& unit)
      : message_(std::string("Some of unit \"") + unit +
                 "\"'s parents did not run.") {
  }

  virtual const char* what() const noexcept {
    return message_.c_str();
  }

 private:
  std::string message_;
};

template<typename... Types>
using variant = mapbox::util::variant<Types...>;

using Property = variant<bool, int, float, std::string, PackagedNumpyArray>;

struct WalkDecision {
  WalkDecision() : value(0) {}
#if 0
  template <class T,
            typename std::enable_if<std::is_integral<T>::value>::type = false>
  /* not explicit */ WalkDecision(T v) : value(v) {}
#endif
  WalkDecision(size_t v) : value(v) {}
  WalkDecision(bool v) : value(v) {}
  WalkDecision(int v) : value(v) {}

  static constexpr int kStop = 0;
  static constexpr int kContinue = 1;
  static constexpr int kIgnore = 2;

  operator bool() const noexcept { return value; }

  int value;
};

/** @brief VELES neural network unit
 */
class Unit : public virtual Logger, public std::enable_shared_from_this<Unit> {
 public:
  Unit(const std::shared_ptr<Engine>& engine);
  virtual ~Unit() = default;
  /** @brief UUID4 which identifies the corresponding VELES unit class.
   */
  virtual const std::string& Uuid() const noexcept = 0;
  /** @brief Sets or modifies a parameter
   *  @param name Parameter name
   *  @param value Parameter value.
   *  @note GetParameter() is useless because the inheriting class should provide
   *  dedicated getters.
   */
  virtual void SetParameter(const std::string& name, const Property& value) = 0;
  /** @brief Returns the needed output size in bytes.
   */
  virtual size_t OutputSize() const = 0;
  /** @brief Performs initial unit initialization. By default, it just calls
   * Reset().
   */
  virtual void Initialize();
  /** @brief Executes this unit, calling child units if they become ready.
   * Internally, invokes Execute() virtual method which does the actual work.
   */
  void Run();
  /**
   * @brief Establishes a link from parent to this unit.
   */
  void LinkFrom(const std::shared_ptr<Unit>& parent);

  const std::vector<std::shared_ptr<Unit>>& Children() const noexcept {
    return links_from_;
  }

  const std::vector<std::weak_ptr<Unit>>& Parents() const noexcept {
    return links_to_;
  }

  void* output() const noexcept {
    return output_;
  }

  void set_output(void* value) noexcept {
    output_ = value;
  }

  bool gate() const noexcept { return gate_; }
  /** @brief Reset the unit, so that it is ready for the next execution.
   */
  virtual void Reset();

  /** Returns true is all parent units have been executed; otherwise, false.
   */
  bool Ready() const noexcept;

  const Unit* BreadthFirstWalk(
      const std::function<WalkDecision(const Unit* unit)>& payload,
      bool include_self=true) const;
  const Unit* DepthFirstWalk(
      const std::function<WalkDecision(const Unit* unit)>& payload,
      bool include_self=true) const;

  const Unit* ReversedBreadthFirstWalk(
      const std::function<WalkDecision(const Unit* unit)>& payload,
      bool include_self=true) const;
  const Unit* ReversedDepthFirstWalk(
      const std::function<WalkDecision(const Unit* unit)>& payload,
      bool include_self=true) const;

 protected:
  virtual void Execute() = 0;

 private:
  std::shared_ptr<Engine> engine_;
  mutable void* output_;
  bool gate_;
  /// this -> children
  std::vector<std::shared_ptr<Unit>> links_from_;
  /// parents -> this
  std::vector<std::weak_ptr<Unit>> links_to_;
};

}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_UNIT_H_
