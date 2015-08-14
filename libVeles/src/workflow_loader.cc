/*! @file workflow_loader.cc
 *  @brief Implementation of WorkflowLoader class.
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

#include "veles/workflow_loader.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <exception>
#include <fstream>
#include <vector>
#include <libarchive/libarchive/archive_entry.h>  // NOLINT(*)
#include <libarchive/libarchive/archive.h>  // NOLINT(*)
#include "inc/veles/unit_factory.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::ifstream;

namespace veles {

/// Name of the file which describes the workflow.
const char* WorkflowLoader::kMainFile = "contents.json";

WorkflowLoader::WorkflowLoader() {
}

Workflow WorkflowLoader::Load(const string& /*archive*/) {
  return Workflow();
}

WorkflowLoader::WorkflowArchive WorkflowLoader::ExtractArchive(
    const string& /*filename*/) {
  return WorkflowArchive();
}

}  // namespace veles
