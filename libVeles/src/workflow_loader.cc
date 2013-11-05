/*! @file workflow_loader.cc
 *  @brief New file description.
 *  @author Bulychev Egor <e.bulychev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "inc/veles/workflow_loader.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <exception>
#include <fstream>
#include <vector>
#include <libarchive/libarchive/archive_entry.h>  // NOLINT(*)
#include <libarchive/libarchive/archive.h>  // NOLINT(*)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <yaml-cpp/yaml.h>  // NOLINT(*)
#pragma GCC diagnostic pop
#include "inc/veles/unit_factory.h"


using std::string;
using std::vector;
using std::shared_ptr;
using std::ifstream;
using std::unique_ptr;
using veles::WorkflowLoader;
using veles::UnitDescription;
using veles::PropertiesTable;
using veles::WorkflowDescription;
using veles::WorkflowExtractionError;

/// Default name of decompressed yaml file.
const char* WorkflowLoader::kWorkflowDecompressedFile =
    "default.yaml";

const char* WorkflowLoader::kDefaultWorkingDirectory = "/tmp/workflow_tmp/";

WorkflowLoader::WorkflowLoader()
    : working_directory_(kDefaultWorkingDirectory) {
}

// TODO(EBulychev): delete fileWithWorkflow. fileWithWorkflow will always be
// workflow.yaml
void WorkflowLoader::Load(const string& archive) {
  archive_name_ = archive;
  file_with_workflow_ = kWorkflowDecompressedFile;

  //  1) Extract archive (using libarchive) to directory working_directory_.
  WorkflowLoader::ExtractArchive(archive_name_, working_directory_);
  DBG("Extracted %s to %s", archive_name_.c_str(), working_directory_.c_str());
  //  2) Read neural network structure from file_with_workflow_
  auto workflow_file = working_directory_ + file_with_workflow_;
  WorkflowLoader::GetWorkflow(workflow_file);
  DBG("Loaded %s", file_with_workflow_.c_str());
  // Remove the working directory with all files
  WorkflowLoader::RemoveDirectory(working_directory_);
}

void WorkflowLoader::GetWorkflow(const string& yaml_filename) {
  vector<YAML::Node> workflow = YAML::LoadAllFromFile(yaml_filename);

  DBG("Number of extracted nodes from yaml: %zu", workflow.size());

  if (workflow.size() == 1) {
    // enum value { Undefined, Null, Scalar, Sequence, Map };
    DBG("Node type %d", workflow.at(0).Type());
    CreateWorkflow(workflow.at(0));
  } else {
    ERR("Can't extract workflow");
    throw std::runtime_error(
        "veles::WorkflowLoader::GetWorkflow: can't extract workflow");
  }
}

void WorkflowLoader::CreateWorkflow(const YAML::Node& doc) {
  for (auto& it : doc) {
    DBG("inside");
    string key, value;

    if (it.first.IsScalar() && it.second.IsScalar()) {
      DBG("Scalar & scalar");
      key = it.first.as<string>();
      value = it.second.as<string>();
      shared_ptr<string> temp(new string(value));

      workflow_desc_.Properties.insert({key, temp});
      key.clear();
      value.clear();
    } else if (it.first.IsScalar() && it.second.IsMap()) {
      DBG("Scalar & map");
      UnitDescription unit;
      unit.Name = it.first.as<string>();
      WorkflowLoader::GetUnit(it.second, &unit);
      WorkflowLoader::workflow_desc_.Units.push_back(unit);
    } else {
      // It can't be neither Scalar nor Map!!!
      ERR("veles::WorkflowLoader::CreateWorkflow: bad YAML::Node");
      throw std::runtime_error(
          "veles::WorkflowLoader::CreateWorkflow: bad YAML::Node");
    }
  }
}

void WorkflowLoader::GetUnit(const YAML::Node& doc, UnitDescription* unit) {
  for (auto& it : doc) {
    string key, value;
    // Add properties to UnitDescription
    if (it.first.IsScalar() && it.second.IsScalar()) {
      // Add properties
      key = it.first.as<string>();
      value = it.second.as<string>();
      auto temp = std::make_shared<string>(value);
      unit->Properties.insert({key, temp});
      // Get array from file
      if (key.find(string("link_to_")) != string::npos) {
          string new_key = key.substr(string("link_to_").size());
          size_t array_size = 0;
          unit->Properties.insert({new_key, GetArrayFromFile(value,
                                                             &array_size)});
          string new_key_to_size = new_key + "_length";
          auto temp_size = std::make_shared<size_t>(array_size);
          unit->Properties.insert({new_key_to_size, temp_size});
      }
    } else if (it.first.IsScalar() && (it.second.IsMap() ||
        it.second.IsSequence())) {
      // Recursive adding properties
      unit->Properties.insert({it.first.as<string>(),
        GetProperties(it.second)});
    } else {
      throw std::runtime_error(
                    "veles::WorkflowLoader::GetUnit: bad YAML::Node");
    }
  }
}

shared_ptr<void> WorkflowLoader::GetProperties(const YAML::Node& node) {
  if (node.IsScalar()) {
    // Simplest variant - return shared_ptr to string or to float array
    auto temp = std::make_shared<string>(node.as<string>());
    return temp;
  } else if (node.IsMap()) {
    auto tempVectorMap = std::make_shared<vector<PropertiesTable>>();
    for (const auto& it : node) {
      PropertiesTable tempProp;
      tempProp.insert({it.first.as<string>(), GetProperties(it.second)});
      tempVectorMap.get()->push_back(tempProp);
    }
    return tempVectorMap;
  } else if (node.IsSequence()) {
    auto tempVectorSequence = std::make_shared<vector<shared_ptr<void>>>();
    for (const auto& it : node) {
      tempVectorSequence->push_back(GetProperties(it));
    }
    return tempVectorSequence;
  }
  ERR("veles::WorkflowLoader::GetProperties: bad YAML::Node");
  throw std::runtime_error
  ("veles::WorkflowLoader::GetProperties: bad YAML::Node");
}

template <typename T>
void delete_array(T* p) {
  delete[] p;
}

float* mallocf(size_t length) {
  void *ptr;
  return posix_memalign(&ptr, 64, length * sizeof(float)) == 0 ?
      static_cast<float*>(ptr) : nullptr;
}

std::shared_ptr<float> WorkflowLoader::GetArrayFromFile(const string& file,
                                                        size_t* arr_size) {
  string link_to_file = working_directory_ + file;
  // Open extracted files
  ifstream fr(link_to_file, std::ios::in | std::ios::binary | std::ios::ate);
  if (!fr.is_open()) {
    throw std::runtime_error
      ("veles::WorkflowLoader::GetArrayFromFile: Can't open file with array");
  }
  // Calculate size of float array to read from file
  fr.seekg (0, fr.end);
  int array_size = fr.tellg();
  fr.seekg (0, fr.beg);

  if (arr_size) {
    *arr_size = array_size / sizeof(float);
  }
  // Read array
  auto array = shared_ptr<float>(mallocf(array_size), free);
  fr.read(reinterpret_cast<char*>(array.get()), array_size);

  DBG("%s size = %d bytes: %f, %f, ...", file.c_str(),
      array_size, array.get()[0], array.get()[1]);

  return array;
}

void WorkflowLoader::InitializeWorkflow() {
  for (auto& it : workflow_desc_.Units) {
    auto unit_constructor = veles::UnitFactory::Instance()[it.Name];
    if (unit_constructor) {
      auto unit = unit_constructor();
      for (auto& it_unit : it.Properties) {
        if (string("bias_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *static_cast<size_t*>(it_unit.second.get()));
        } else if (string("input_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *static_cast<size_t*>(it_unit.second.get()));
        } else if (string("output_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *static_cast<size_t*>(it_unit.second.get()));
        } else if (string("weights_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *static_cast<size_t*>(it_unit.second.get()));
        } else {
          INF("Properties: %s", it_unit.first.c_str());
        }
        unit->SetParameter(it_unit.first, it_unit.second);
      }
      workflow_.Add(unit);
    }
  }
}

veles::Workflow WorkflowLoader::GetWorkflow() {
  if (workflow_.Size() == 0U) {
    InitializeWorkflow();
  }
  return workflow_;
}

string WorkflowLoader::PrintWorkflowStructure() {
  // Workflow properties to string
  string result = "\nWorkFlow properties:\n";

  for (auto x : workflow_desc_.Properties) {
    result += x.first + " : " + *(static_cast<string*>(x.second.get())) + "\n";
  }
  // Print units and their properties
  for (unsigned i = 0; i < workflow_desc_.Units.size(); ++i) {
    result += "\nUnit name: " + workflow_desc_.Units.at(i).Name + "\n";

    for (auto y : workflow_desc_.Units.at(i).Properties) {
      result += y.first + " : " + *static_cast<string*>(y.second.get()) + "\n";
    }
  }
  return result;
}

void WorkflowLoader::ExtractArchive(const string& filename,
                                    const string& directory) {
  static const size_t kBlockSize = 10240;

  auto destroy_read_archive = [](archive* ptr) {
    archive_read_close(ptr);
    archive_read_free(ptr);
  };

  auto input_archive = std::uniquify(archive_read_new(), destroy_read_archive);
  auto destroy_write_archive = [](archive* ptr) {
    archive_write_close(ptr);
    archive_write_free(ptr);
  };
  auto ext = std::uniquify(archive_write_disk_new(), destroy_write_archive);

  archive_entry *entry;
  int r;

  try {
    archive_read_support_filter_all(input_archive.get());
    archive_read_support_format_all(input_archive.get());
    archive_read_support_format_tar(input_archive.get());
    if ((r = archive_read_open_filename(input_archive.get(), filename.c_str(),
                                        kBlockSize))) {
      ERR("archive_read_open_filename(): %s",
          archive_error_string(input_archive.get()));
      throw WorkflowExtractionFailedException(filename, archive_error_string(
          input_archive.get()));
    }

    while ((r = archive_read_next_header(input_archive.get(), &entry) !=
        ARCHIVE_EOF)) {
      if (r != ARCHIVE_OK) {
        auto error = archive_error_string(input_archive.get());
        if (error != nullptr) {
          ERR("archive_read_next_header() : %s", error);
          throw WorkflowExtractionFailedException(filename, error);
        }
      }
      auto path = directory + "/" + archive_entry_pathname(entry);

      archive_entry_set_pathname(entry, path.c_str());

      r = archive_write_header(ext.get(), entry);
      if (r != ARCHIVE_OK) {
        DBG("archive_write_header() : %s",
            archive_error_string(ext.get()));
      } else {
        CopyData(*input_archive.get(), ext.get());
        r = archive_write_finish_entry(ext.get());
        if (r != ARCHIVE_OK) {
          DBG("archive_write_finish_entry() : %s",
              archive_error_string(ext.get()));
        }
      }
    }
  } catch(const std::exception& e) {
    auto error = string("(veles::WorkflowLoader::ExtractArchive:\nCan't open "
        "archive: ") + e.what() + ")";
    ERR("%s", error.c_str());
    throw std::runtime_error(error);
  }
}

void WorkflowLoader::RemoveDirectory(const string& path) {
  {
    auto dir = std::uniquify(opendir(path.c_str()), closedir);

    if (!dir) {  // if dir wasn't initialized correctly
      throw std::runtime_error
        ("Can't open directory to delete, check path + permissions\n");
    }  // end if

    dirent *pent = nullptr;

    unsigned char isFile = DT_REG;  // magic number to find files in directory
    while ((pent = readdir(dir.get()))) {
      // while there is still something in the directory to list
      if (pent->d_type == isFile) {
        string path_to_del = (path + "/" + pent->d_name);

        remove(path_to_del.c_str());
      }
    }
  }
  if (rmdir(path.c_str()) != 0) {
    throw std::runtime_error
            ("Can't delete directory, check path + permissions\n");
  }
}

int WorkflowLoader::CopyData(const archive& ar, archive *aw) {
  int res;
  const void *buff;
  size_t size;
  int64_t offset;
  do {
    res = archive_read_data_block(const_cast<archive*>(&ar), &buff, &size,
                                  &offset);
    if (res == ARCHIVE_EOF) {
     return ARCHIVE_OK;
    }
    if (res != ARCHIVE_OK) {
     ERR("From CopyData : archive_write_data_block() : %s",
         archive_error_string(aw));
     break;
    }
    res = archive_write_data_block(aw, buff, size, offset);
  } while (res == ARCHIVE_OK);

  return res;
}

const WorkflowDescription& WorkflowLoader::workflow_desc() const {
  return workflow_desc_;
}
