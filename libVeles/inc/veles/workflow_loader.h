/*! @file workflow_loader.h
 *  @brief Declaration of WorkflowLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>, Bulychev Egor <e.bulychev@samsung.com>
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

#ifndef INC_VELES_WORKFLOW_LOADER_H_
#define INC_VELES_WORKFLOW_LOADER_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <veles/logger.h>  // NOLINT(*)
#include <veles/workflow.h>  // NOLINT(*)
#include <veles/poison.h>  // NOLINT(*)

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

class WorkflowLoadingFailedException : public std::exception {
 public:
  WorkflowLoadingFailedException(const std::string& file,
                                 const std::string& reason)
      : message_(std::string("Extraction of the workflow \"") + file +
                 "\" has failed due to " + reason + ".") {
  }

  virtual const char* what() const noexcept {
    return message_.c_str();
  }

 private:
  std::string message_;
};

class WorkflowArchive;

/**
 * @brief Factory which produces Workflow objects from packages stored on disk.
 * */
class WorkflowLoader : protected DefaultLogger<WorkflowLoader,
                                               Logger::COLOR_YELLOW> {
 public:
  WorkflowLoader();
  virtual ~WorkflowLoader() = default;
  /// @brief Main function.
  /**
   * @param[in] archive Path to the package.
   * @return The loaded and ready to be initialized Workflow instance.
   */
  Workflow Load(const std::string& archive);

 private:
  friend class WorkflowLoaderTest;
};

}  // namespace Veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_WORKFLOW_LOADER_H_
