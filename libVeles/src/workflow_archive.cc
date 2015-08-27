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
#include <cstring>
#include <list>
#include <libarchive/libarchive/archive.h>  // NOLINT(*)
#include <libarchive/libarchive/archive_entry.h>  // NOLINT(*)
#include "inc/veles/make_unique.h"
#include "src/iarchivestream.h"
#include "src/imemstream.h"
#include "src/main_file_loader.h"

namespace veles {

namespace internal {

/// Name of the file which describes the workflow.
const char* WorkflowArchive::kMainFile = "contents.json";
const Logger WorkflowArchive::kLogger( "WorkflowArchive", EINA_COLOR_YELLOW);

WorkflowArchive::WorkflowArchive(const WorkflowDefinition& wdef)
    : workflow_definition_(wdef) {
}

std::shared_ptr<WorkflowArchive> WorkflowArchive::Load(
    const std::string& file_name) {
  int error = ARCHIVE_OK;
  auto arch = Open(file_name, &error);
  if (!arch || error != ARCHIVE_OK) {
    ERRI(&kLogger, "Failed to open %s: %d", file_name.c_str(), error);
    return nullptr;
  }
  bool isTar = archive_filter_code(arch.get(), 0) != ARCHIVE_FILTER_NONE;
  DBGI(&kLogger, "Successfully opened %s (TAR: %s), scanning for %s",
       file_name.c_str(), isTar? "true" : "false", kMainFile);
  archive_entry* entry;
  std::shared_ptr<WorkflowArchive> wa;
  std::unordered_map<std::string, shared_array<uint8_t>> files;
  while (archive_read_next_header(arch.get(), &entry) == ARCHIVE_OK) {
    std::string efn = archive_entry_pathname(entry);
    if (efn == kMainFile) {
      DBGI(&kLogger, "Found %s", kMainFile);
      auto instr = std::make_unique<iarchivestream>(arch);
      auto wdef = MainFileLoader().Load(instr.get());
      wa = std::make_shared<WorkflowArchive>(wdef);
    } else if (!isTar) {
      DBGI(&kLogger, "Skipping %s...", efn.c_str());
      archive_read_data_skip(arch.get());
    } else {
      auto size = archive_entry_size(entry);
      shared_array<uint8_t> mem;
      if (size > 0) {
        DBGI(&kLogger, "Reading %s (%d bytes)...", efn.c_str(),
             static_cast<int>(size));
        mem.reset(new uint8_t[size], size);
        archive_read_data(arch.get(), mem.get_raw(), size);
      } else {
        std::list<shared_array<uint8_t>> chunks;
        ssize_t read_size;
        const int BUFFER_SIZE = UINT16_MAX + 1;
        shared_array<uint8_t> buf(new uint8_t[BUFFER_SIZE], BUFFER_SIZE);
        while ((read_size = archive_read_data(
            arch.get(), buf.get_raw(), BUFFER_SIZE)) > 0) {
          buf.set_size(read_size);
          chunks.push_back(buf);
          if (read_size == BUFFER_SIZE) {
            buf.reset(new uint8_t[BUFFER_SIZE], BUFFER_SIZE);
          } else {
            buf.reset();
          }
        }
        for (auto& arr : chunks) {
          size += arr.size();
        }
        DBGI(&kLogger, "Read %s (%d bytes in %d chunks)...", efn.c_str(),
             static_cast<int>(size), static_cast<int>(chunks.size()));
        mem.reset(new uint8_t[size], size);
        auto head = mem.get_raw();
        for (auto& arr : chunks) {
          memcpy(head, arr.get_raw(), arr.size());
          head += arr.size();
        }
      }
      files[efn] = mem;
    }
  }
  wa->file_name_ = file_name;
  wa->files_ = files;
  return wa;
}

const WorkflowDefinition& WorkflowArchive::workflow_definition() const {
  return workflow_definition_;
}

const std::string& WorkflowArchive::file_name() const {
  return file_name_;
}

std::shared_ptr<std::istream> WorkflowArchive::GetStream(
    const std::string& file_name) const {
  auto it = files_.find(file_name);
  if (it != files_.end()) {
    return std::make_shared<imemstream<uint8_t>>(it->second);
  }
  int error = ARCHIVE_OK;
  auto arch = Open(file_name_, &error);
  if (!arch || error != ARCHIVE_OK) {
    ERR("Failed to open %s: %d", file_name.c_str(), error);
    return nullptr;
  }
  archive_entry* entry;
  while (archive_read_next_header(arch.get(), &entry) == ARCHIVE_OK) {
    std::string efn = archive_entry_pathname(entry);
    if (efn == file_name) {
      return std::make_shared<iarchivestream>(arch);
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

}  // namespace internal

}  // namespace veles
