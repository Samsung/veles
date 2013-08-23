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
#include <fstream>
#include <vector>
#include <exception>
#include "inc/veles/make_unique.h"


using std::string;
using std::vector;
using std::shared_ptr;
using std::ifstream;
using std::unique_ptr;
using Veles::WorkflowLoader;
using Veles::UnitDescription;
using Veles::PropertiesTable;
using Veles::WorkflowDescription;
using Veles::WorkflowExtractionError;

// TODO(EBulychev): Add array reading for recursion + test workability

const char* WorkflowLoader::kWorkFolder = "/tmp/workflow_tmp/";
/// Default name of decompressed yaml file.
const char* WorkflowLoader::kWorkflowDecompressedFile =
    "workflow_decompressed.yaml";

void WorkflowLoader::Load(const string& archive,
                         const string& fileWithWorkflow) {
  archive_name_ = archive;
  file_with_workflow_ = fileWithWorkflow;
//  1) Extract archive (using libarchive) to folder kWorkFolder.
  WorkflowLoader::ExtractArchive(archive.c_str());

//  2) Read neural network structure from fileWithWorkflow
  string workflow_file = kWorkFolder + file_with_workflow_;
  WorkflowLoader::GetWorkflow(workflow_file);

  // Remove kWorkFolder with all files
  WorkflowLoader::RemoveDirectory(kWorkFolder);
}

void WorkflowLoader::GetWorkflow(const string& yaml_filename) {
  vector<YAML::Node> workflow = YAML::LoadAllFromFile(yaml_filename);

  if (workflow.size() == 1) {
    CreateWorkflow(workflow.at(0));
    PrintWorkflowStructure();
  } else {
    throw std::runtime_error(
              "Veles::WorkflowLoader::GetWorkflow: bad YAML::Node");
  }
}

void WorkflowLoader::CreateWorkflow(const YAML::Node& doc) {
  for (auto& it : doc) {
    string key, value;

    if (it.first.IsScalar() && it.second.IsScalar()) {
      key = it.first.as<string>();
      value = it.second.as<string>();
      shared_ptr<string> temp(new string(value));

      workflow_.Properties.insert({key, temp});
      key.clear();
      value.clear();
    } else if (it.first.IsScalar() && it.second.IsMap()) {
      UnitDescription unit;
      unit.Name = it.first.as<string>();
      WorkflowLoader::GetUnit(it.second, &unit);
      WorkflowLoader::workflow_.Units.push_back(unit);
    } else {
      // It can't be neither Scalar nor Map!!!
      throw std::runtime_error(
          "Veles::WorkflowLoader::CreateWorkflow: bad YAML::Node");
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
          unit->Properties.insert({new_key, GetArrayFromFile(value)});
      }
    } else if (it.first.IsScalar() && (it.second.IsMap() ||
        it.second.IsSequence())) {
      // Recursive adding properties
      unit->Properties.insert({it.first.as<string>(),
        GetProperties(it.second)});
    } else {
      throw std::runtime_error(
                    "Veles::WorkflowLoader::GetUnit: bad YAML::Node");
    }
  }
}

shared_ptr<void> WorkflowLoader::GetProperties(const YAML::Node& node) {
  if (node.IsScalar()) {
    // Simplest variant - return shared_ptr to string or to float array
    auto temp = std::make_shared<string>(node.as<string>());
    return temp;
  } else if (node.IsMap()) {
    shared_ptr<vector<PropertiesTable>> tempVectorMap;
    for (const auto& it : node) {
      PropertiesTable tempProp;
      tempProp.insert({it.first.as<string>(), GetProperties(it.second)});
      tempVectorMap.get()->push_back(tempProp);
    }
    return tempVectorMap;
  } else if (node.IsSequence()) {
    shared_ptr<vector<shared_ptr<void>>> tempVectorSequence;
    for (const auto& it : node) {
      tempVectorSequence.get()->push_back(GetProperties(it));
    }
    return tempVectorSequence;
  }
  throw std::runtime_error
  ("Veles::WorkflowLoader::GetProperties: bad YAML::Node");
}

template <typename T>
void delete_array(T* p) {
  delete[] p;
}

shared_ptr<float>
WorkflowLoader::GetArrayFromFile(const string& file) {
  string link_to_file = kWorkFolder + file;
  // Open extracted files
  ifstream fstream(link_to_file, std::ios::in|std::ios::binary|
                   std::ios::ate);
  // Calculate size of float array to read from file
  int array_size = fstream.tellg();
  // Read array
  auto weight = shared_ptr<float>(new float[array_size],
                                       delete_array<float>);
  fstream.read(reinterpret_cast<char*>(weight.get()), array_size*sizeof(float));

  return weight;
}

string WorkflowLoader::PrintWorkflowStructure() {
  // Workflow properties to string
  string result = "\nWorkFlow properties:\n";

  for (auto x : workflow_.Properties) {
    result += x.first + " : " + *(static_cast<string*>(x.second.get())) + "\n";
  }
  // Print units and their properties
  for (unsigned i = 0; i < workflow_.Units.size(); ++i) {
    result += "\nUnit name: " + workflow_.Units.at(i).Name + "\n";

    for (auto y : workflow_.Units.at(i).Properties) {
      result += y.first + " : " + *static_cast<string*>(y.second.get()) + "\n";
    }
  }
  return result;
}

void WorkflowLoader::ExtractArchive(const string& filename,
    const string& folder) {
  // Check that folder ends with '/'
  char ch = folder.back();
  string folder2;
  if (ch != '/') {
    folder2 = folder + string("/");
  } else {
    folder2 = folder;
  }
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
      string error =
          string(R"(Veles::WorkflowLoader::ExtractArchive:
 archive_read_open_filename() : )") +
          string(archive_error_string(input_archive.get()));
      throw std::runtime_error(error);
    }

    while ((r = archive_read_next_header(input_archive.get(), &entry) !=
        ARCHIVE_EOF)) {
      if (r != ARCHIVE_OK) {
        fprintf(stderr, "archive_read_next_header() : %s\n",
               archive_error_string(input_archive.get()));
      }
      auto path = folder2 + archive_entry_pathname(entry);

      archive_entry_set_pathname(entry, path.c_str());

      r = archive_write_header(ext.get(), entry);
      if (r != ARCHIVE_OK) {
        fprintf(stderr, "archive_write_header() : %s\n",
               archive_error_string(ext.get()));
      } else {
        CopyData(*input_archive.get(), ext.get());
        r = archive_write_finish_entry(ext.get());
        if (r != ARCHIVE_OK) {
          fprintf(stderr, "archive_write_finish_entry() : %s\n",
                         archive_error_string(ext.get()));
        }
      }
    }
  } catch(const std::exception& e) {
    string error =
              string(R"(Veles::WorkflowLoader::ExtractArchive:
Can't open archive : )") + string(e.what());
          throw std::runtime_error(error);
  }
}

void WorkflowLoader::RemoveDirectory(const string& path) {
  {
  auto dir = std::uniquify(opendir(path.c_str()), closedir);

    if (!dir) {  // if pdir wasn't initialised correctly
      throw std::runtime_error
        ("Can't open directory to delete, check path + permissions\n");
    }  // end if

    dirent *pent = nullptr;

    unsigned char isFile = DT_REG;  // magic number to find files in folder
    while ((pent = readdir(dir.get()))) {
      // while there is still something in the directory to list
      if (pent->d_type == isFile) {
        const char *file_to_delete;
        file_to_delete = string(path + "/" + string(pent->d_name)).c_str();
        remove(file_to_delete);
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
     fprintf(stderr, "From CopyData : archive_write_data_block() : %s\n",
             archive_error_string(aw));
     break;
    }
    res = archive_write_data_block(aw, buff, size, offset);
  } while (res == ARCHIVE_OK);

  return res;
}
