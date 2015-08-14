/*! @file workflow_loader.cc
 *  @brief Implementation of WorkflowLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>, Bulychev Egor <e.bulychev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
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
