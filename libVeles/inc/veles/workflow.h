/*! @file workflow.h
 *  @brief VELES Workflow class declaration.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013, 2015 Samsung R&D Institute Russia
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

#ifndef INC_VELES_WORKFLOW_H_
#define INC_VELES_WORKFLOW_H_

#include <memory>
#include <string>
#include <veles/logger.h>
#include <veles/unit.h>
#include <veles/make_unique.h>
#include <veles/memory_node.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

class ParentUnitDoesNotExistException : public std::exception {
 public:
  ParentUnitDoesNotExistException(const std::string& unit)
      : message_(std::string("Parent unit \"") + unit + "\" does not exist.") {
  }

  virtual const char* what() const noexcept {
    return message_.c_str();
  }

 private:
  std::string message_;
};

/** @brief VELES workflow */
class Workflow : protected DefaultLogger<Workflow, Logger::COLOR_ORANGE> {
 public:
  explicit Workflow(const std::string& name, const std::string& checksum,
                    const std::shared_ptr<Unit>& head,
                    const std::shared_ptr<Engine>& engine);
  virtual ~Workflow() = default;
  const std::string& name() const noexcept { return name_; }
  const std::string& checksum() const noexcept { return checksum_; }

  /** @brief Clears the Workflow
   */
  void Clear() noexcept;

  /** @brief Number of units
   */
  size_t Size() const noexcept;

  std::shared_ptr<Unit> Tail() const noexcept;

  std::shared_ptr<Unit> Head() const noexcept { return head_; }

  /** @brief Prepares the workflow for execution.
   *  @param input The input data for the first unit.
   */
  void Initialize(const void* input);

  void Run();

  /** @brief Returns the output from the last unit. The pointer is guaranteed
   *  to not change before the next call to Initialize().
   */
  void* output() const noexcept {
    return Tail()->output();
  }

  static void* malloc_aligned_void(size_t size);
  template <class T>
  static T* malloc_aligned(size_t length) {
    return reinterpret_cast<T*>(malloc_aligned_void(length * sizeof(T)));
  }

 private:
  std::vector<internal::MemoryNode> StateMemoryOptimizationProblem() const;

  std::string name_;
  std::string checksum_;
  std::shared_ptr<Unit> head_;
  std::shared_ptr<Engine> engine_;
  std::shared_ptr<uint8_t> boilerplate_;
};

}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_WORKFLOW_H_
