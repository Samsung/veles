/*! @file workflow_archive.h
 *  @brief Helper class to deal with workflow archives (packages).
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2015 Samsung R&D Institute Russia
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

#ifndef SRC_WORKFLOW_ARCHIVE_H_
#define SRC_WORKFLOW_ARCHIVE_H_

#include <string>
#include <unordered_map>
#include "src/main_file_loader.h"
#include "src/shared_array.h"

struct archive;

namespace veles {

class WorkflowArchive : protected DefaultLogger<WorkflowArchive,
                                                Logger::COLOR_YELLOW> {
 public:
  explicit WorkflowArchive(const WorkflowDefinition& wdef);
  virtual ~WorkflowArchive() = default;

  const WorkflowDefinition& workflow_definition() const;
  const std::string& file_name() const;

  static std::shared_ptr<WorkflowArchive> Load(const std::string& file_name);

  std::shared_ptr<std::istream> GetStream(const std::string& file_name) const;

 private:
  friend class WorkflowArchiveTest_MnistWorkflow_Test;
  /// @brief Initializes a new instance of struct archive (from libarchive).
  /// @param file_name The path to the archive to open.
  /// @param error Used for reporting back the errors (may be nullptr).
  /// @return Shared pointer to struct archive with the proper destructor.
  static std::shared_ptr<archive> Open(const std::string& file_name, int* error);

  /// Name of the file which describes the workflow.
  static const char* kMainFile;
  static const Logger kLogger;

  WorkflowDefinition workflow_definition_;
  std::string file_name_;
  std::unordered_map<std::string, shared_array<uint8_t>> files_;
};

}  // namespace veles
#endif  // SRC_WORKFLOW_ARCHIVE_H_
