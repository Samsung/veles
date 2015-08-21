/*! @file workflow_archive.cc
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

#include "src/workflow_archive.h"
#include <cassert>
#include <libarchive/libarchive/archive.h>  // NOLINT(*)
#include <libarchive/libarchive/archive_entry.h>  // NOLINT(*)
#include "src/iarchivestream.h"

namespace veles {

WorkflowArchive::WorkflowArchive(const WorkflowDefinition& wdef)
    : workflow_definition_(wdef) {
}

const WorkflowDefinition& WorkflowArchive::workflow_definition() const {
  return workflow_definition_;
}

std::shared_ptr<std::istream> WorkflowArchive::GetNumpyArrayStream(
    const std::string& file_name) const {
  int error;
  auto arch = Open(file_name, &error);
  assert(arch);
  archive_entry* entry;
  while (archive_read_next_header(arch.get(), &entry) == ARCHIVE_OK) {
    std::string efn = archive_entry_pathname(entry);
    if (efn == file_name) {
      return std::make_shared<iarchivestream>(arch.get());
    } else {
      archive_read_data_skip(arch.get());
    }
  }
  return nullptr;
}

std::shared_ptr<archive> WorkflowArchive::Open(
    const std::string& file_name, int* error) {
  auto arch = std::shared_ptr<archive>(archive_read_new(), archive_read_free);
  assert(archive_read_support_filter_all(arch.get()) == ARCHIVE_OK);
  assert(archive_read_support_format_tar(arch.get()) == ARCHIVE_OK);
  assert(archive_read_support_format_zip(arch.get()) == ARCHIVE_OK);
  int res = archive_read_open_filename(
      arch.get(), file_name.c_str(), UINT16_MAX + 1);
  if (res != ARCHIVE_OK) {
    if (error) {
      *error = res;
    }
    return nullptr;
  }
  return std::move(arch);
}

}  // namespace veles
