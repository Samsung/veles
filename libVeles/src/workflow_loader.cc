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
#include <veles/workflow_loader.h>
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
const char* WorkflowLoader::kWorkflowDecompressedFile = "default.yaml";
/// Default path for archive's decompressing.
const char* WorkflowLoader::kWorkingDirectory = "/tmp/workflow_tmp";

WorkflowLoader::WorkflowLoader() : working_directory_path_(kWorkingDirectory) {
}

void WorkflowLoader::Load(const string& archive) {
  //  1) Extract archive (using libarchive) to directory kWorkingDirectory.
  WorkflowLoader::ExtractArchive(archive);
  DBG("Successful archive extracting\n");
  //  2) Read neural network structure from fileWithWorkflow
  auto workflow_file = working_directory_path_;
  workflow_file /= boost::filesystem::path(kWorkflowDecompressedFile);
//  *workflow_file /= file_with_workflow_;
  DBG("File with workflow: %s\n", workflow_file.string().c_str());
  WorkflowLoader::GetWorkflow(workflow_file.string());
  DBG("Successful reading workflow from yaml\n");
  // Remove the working directory with all files
  WorkflowLoader::RemoveDirectory(working_directory_path_.string());
  DBG("Successful removing directory\n");
}

void WorkflowLoader::GetWorkflow(const string& yaml_filename) {
  vector<YAML::Node> workflow = YAML::LoadAllFromFile(yaml_filename);

  DBG("Number of extracted nodes from yaml: %zu", workflow.size());

  if (workflow.size() == 1) {
    // enum value { Undefined, Null, Scalar, Sequence, Map };
    DBG("Node type %d", workflow.at(0).Type());
    CreateWorkflow(workflow.at(0));
  } else {
    ERR("veles::WorkflowLoader::GetWorkflow: can't extract workflow");
    throw std::runtime_error(
        "veles::WorkflowLoader::GetWorkflow: can't extract workflow");
  }
}

void WorkflowLoader::CreateWorkflow(const YAML::Node& doc) {
  for (auto& it : doc) {
    DBG("inside");
    string key, value;

    if (it.first.IsScalar() && it.second.IsScalar()) {
      DBG("Scalar & scalar : %s & %s", it.first.as<string>().c_str(),
                                       it.second.as<string>().c_str());
      key = it.first.as<string>();
      value = it.second.as<string>();
      shared_ptr<string> temp(new string(value));

      workflow_desc_.Properties.insert({key, temp});
      key.clear();
      value.clear();
    } else if (it.first.IsScalar() && it.second.IsMap()) {
      DBG("Scalar & map : %s & map", it.first.as<string>().c_str());
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
      DBG("GetUnit: Scalar & scalar : %s & %s", it.first.as<string>().c_str(),
                                                it.second.as<string>().c_str());
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
      DBG("GetUnit: Scalar & map(sequence) : %s & map(sequence)",
          it.first.as<string>().c_str());
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
    DBG("GetProperties: Scalar : %s", node.as<string>().c_str());
    auto temp = std::make_shared<string>(node.as<string>());
    return temp;
  } else if (node.IsMap()) {
    auto props_map = std::make_shared<vector<PropertiesTable>>();
    for (const auto& it : node) {
      DBG("GetProperties: Map : %s & second", it.first.as<string>().c_str());
      PropertiesTable props;
      props.insert({it.first.as<string>(), GetProperties(it.second)});
      props_map.get()->push_back(props);
    }
    return props_map;
  } else if (node.IsSequence()) {
    auto props_sequence = std::make_shared<vector<shared_ptr<void>>>();
    for (const auto& it : node) {
      props_sequence->push_back(GetProperties(it));
    }
    return props_sequence;
  }
  ERR("veles::WorkflowLoader::GetProperties: bad YAML::Node");
  throw std::runtime_error
  ("veles::WorkflowLoader::GetProperties: bad YAML::Node");
}

std::shared_ptr<float> WorkflowLoader::GetArrayFromFile(const string& file,
                                                        size_t* arr_size) {
  auto link_to_file = working_directory_path_;
  link_to_file /= file;
  // Open extracted files
  ifstream fr(link_to_file.string(), std::ios::in | std::ios::binary |
              std::ios::ate);
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
  auto array = shared_ptr<float>(Workflow::mallocf(array_size), free);
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
              *std::static_pointer_cast<const size_t>(it_unit.second));
        } else if (string("input_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *std::static_pointer_cast<const size_t>(it_unit.second));
        } else if (string("output_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *std::static_pointer_cast<const size_t>(it_unit.second));
        } else if (string("weights_length") == it_unit.first) {
          INF("Properties: %s %zu", it_unit.first.c_str(),
              *std::static_pointer_cast<const size_t>(it_unit.second));
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

void WorkflowLoader::ExtractArchive(const string& filename,
                                    const boost::filesystem::path& directory) {
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
        DBG("archive_read_next_header() : %s", error);
      }
      auto path = directory;
      path /= archive_entry_pathname(entry);
      DBG("Extracted file: %s", path.c_str());
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

void WorkflowLoader::RemoveDirectory(const boost::filesystem::path& path) {
  auto tmp_path = path;
  {
    DBG("Try to delete %s", tmp_path.c_str());
    auto dir = std::uniquify(opendir(tmp_path.c_str()), closedir);

    if (!dir) {  // if dir wasn't initialized correctly
      ERR("Can't open directory to delete (%s),\ncheck path + permissions",
          tmp_path.c_str());
      throw std::runtime_error
        ("Can't open directory to delete, check path + permissions");
    }  // end if

    dirent *pent = nullptr;

    unsigned char isFile = DT_REG;  // magic number to find files in directory
    while ((pent = readdir(dir.get()))) {
      // while there is still something in the directory to list
      if (pent->d_type == isFile) {
        auto path_to_del = tmp_path;
        path_to_del /= pent->d_name;
        DBG("Try to delete file: %s", path_to_del.c_str());

        remove(path_to_del.c_str());
      } else if (pent->d_type == DT_DIR){
        if (!strcmp(pent->d_name, ".") || !strcmp(pent->d_name, "..")) {
           continue;
        }
        auto path_to_del = tmp_path;
        path_to_del /= pent->d_name;
        RemoveDirectory(path_to_del);
      }
    }
  }
  if (rmdir(tmp_path.c_str()) != 0) {
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

void WorkflowLoader::set_working_directory(const std::string& directory) {
  working_directory_path_ = directory;
}

const std::string& WorkflowLoader::working_directory() {
  return working_directory_path_.string();
}

void WorkflowLoader::InitWorkflow() {
  auto temp = working_directory_path_;
  temp /= kWorkflowDecompressedFile;
  DBG("InitWorkflow path to workflow file: %s", temp.c_str());
  GetWorkflow(temp.string());
}

void WorkflowLoader::ExtractArchive(const std::string& filename) {
  ExtractArchive(filename, working_directory_path_);
}
