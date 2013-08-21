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

#include <dirent.h>
#include <stdio.h>
#include "fstream"
#include <string.h>
#include <vector>
#include "inc/veles/workflow_loader.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::ifstream;
using Veles::WorkflowLoader;
using Veles::UnitDescription;
using Veles::PropertiesTable;
using Veles::WorkflowDescription;
using Veles::WorkflowExtractionError;

// TODO(EBulychev): unique_ptr, exception, yaml-cpp

const char* WorkflowLoader::kWorkFolder = "/tmp/workflow_tmp/";
/// Default name of decompressed yaml file.
const char* WorkflowLoader::kWorkflowDecompressedFile =
    "workflow_decompressed.yaml";

int WorkflowLoader::Load(const string& archive,
                         const string& fileWithWorkflow) {
  archive_name_ = archive;
  file_with_workflow_ = fileWithWorkflow;
//  1) Extract archive (using libarchive) to folder kWorkFolder.
  if (!WorkflowLoader::ExtractArchive(archive.c_str())) {
    fprintf(stderr, "Error with archive extracting: %s\n", archive.c_str());
    return kArchiveExtractionError;
  }

//  2) Read neural network structure from fileWithWorkflow
  string workflow_file = kWorkFolder + file_with_workflow_;
  if (!WorkflowLoader::GetWorkflow(workflow_file)) {
    string temp = string(kWorkFolder) + string(kWorkflowDecompressedFile);
    fprintf(stderr, "Can't extract workflow from YAML file: %s\n",
      temp.c_str());
    return kWorkflowFromFileExtractionError;
  }
  // Remove kWorkFolder with all files

  if (WorkflowLoader::RemoveDirectory(kWorkFolder)) {
//    printf("Successful folder deleting %s\n", string(kWorkFolder).c_str());
  } else {
    fprintf(stderr, "Unsuccessful folder deleting %s\n", string(kWorkFolder).c_str());
    return kDeletingTempFolderError;
  }
  return kAllGood;
}

bool WorkflowLoader::GetWorkflow(const string& yaml_filename) {
//  string temp = "/home/ebulychev/workspace/compressed_workflow/default.yaml";
//  vector<YAML::Node> workflow = YAML::LoadAllFromFile(temp);
//  printf("Number of readen nodes: %ld\n", workflow.size());
//  printf("workflow is Null : %s\n", (workflow.at(0).IsNull())?"true":"false");
//  printf("workflow is Scalar : %s\n",
//         (workflow.at(0).IsScalar())?"true":"false");
//  printf("workflow is Sequence : %s\n",
//         (workflow.at(0).IsSequence())?"true":"false");
//  printf("workflow is Map : %s\n", (workflow.at(0).IsMap())?"true":"false");
//
//  if (workflow.size() == 1) {
//    CreateWorkflow(workflow.at(0));
//    PrintWorkflowStructure();
//  }

//  YAML::Node lineup = YAML::Load("{1B: Prince Fielder, "
//      "2B: Rickie Weeks, LF: Ryan Braun}");
//
//  for ( YAML::const_iterator it=lineup.begin(); it!=lineup.end(); ++it ) {
//    printf("!!Playing at %s is %s ; Scalar : %s\n",
//           it->first.as<std::string>().c_str(),
//           it->second.as<std::string>().c_str(),
//           (it->second.IsScalar())?"true":"false");
//  }
//  string temp("/home/ebulychev/monsters.yaml");
//  vector<YAML::Node> monsters = YAML::LoadAllFromFile(temp);
//
//  printf("Number of readen nodes: %ld\n", monsters.size());
//  for ( unsigned long i = 0; i < monsters.size(); ++i) {
//    printf("Monster is Null : %s\n", (monsters.at(i).IsNull())?"true":"false");
//    printf("Monster is Scalar : %s\n",
//           (monsters.at(i).IsScalar())?"true":"false");
//    printf("Monster is Sequence : %s\n",
//           (monsters.at(i).IsSequence())?"true":"false");
//    printf("Monster is Map : %s\n", (monsters.at(i).IsMap())?"true":"false");
//
//    if (monsters.at(i).IsSequence()) {
//      printf("CreateWorkflow: %ld\n", monsters.at(i).size());
//    }

//    printf("CreateWorkflow_GetUnit\n");
//    for(YAML::iterator it = monsters.at(i).begin();
//        it!=monsters.at(i).end(); ++it) {
//      printf("Null : %s", (it->second.IsNull())?"true":"false");
//      printf("Scalar : %s", (it->second.IsScalar())?"true":"false");
//      printf("Sequence : %s", (it->second.IsSequence())?"true":"false");
//      printf("Map : %s", (it->second.IsMap())?"true":"false");
//    }
//  for(YAML::const_iterator it=monsters.begin();it!=monsters.end();++it) {
//    if (it->second.IsScalar() == true) {
//      printf("222 Playing at %s is %s : %s\n",
//             it->first.as<std::string>().c_str(),
//             it->second.as<std::string>().c_str(),
//             (it->second.IsScalar())?"true":"false");
//    }
//    else {
//      printf("Null : %s", (it->second.IsNull())?"true":"false")
//      printf("Scalar : %s", (it->second.IsScalar())?"true":"false");
//      printf("Sequence : %s", (it->second.IsSequence())?"true":"false");
//      printf("Map : %s"(it->second.IsMap())?"true":"false");
//    }
//  // Open decompressed yaml file /home/ebulychev/monsters.yaml
//  ifstream fin(yaml_filename);
//  if (!fin) {
//    printf("Can't open file: %s\n", yaml_filename.c_str());
//    return false;
//  }
//  try {
//    YAML::Parser parser(fin);
//    YAML::Node doc;    // already parsed
//    parser.GetNextDocument(doc);
//    WorkflowLoader::CreateWorkflow(doc);
//  } catch(YAML::ParserException& e) {
//    printf("%s\n", e.what());
//    fin.close();
//    return false;
//  }
//  fin.close();

  return true;
}

//void IterateThroughYAML(YAML::Node& node) {
//
//  for (auto it=node.begin();it!=node.end();++it) {
//
//    switch (it->Type()) {
//      case YAML::NodeType::Null:
//        printf("Null;\n");
//        break;
//      case YAML::NodeType::Undefined:
//        printf("Undefined;\n");
//        break;
//      case YAML::NodeType::Scalar:
//        printf("Scalar \n");
//        break;
//      case YAML::NodeType::Sequence:
//        printf("Sequence;\n");
//        break;
//      case YAML::NodeType::Map:
//        printf("Map \n");
//
//        break;
//    }
//  }
//}

bool WorkflowLoader::CreateWorkflow(const YAML::Node& doc) {
//  for (auto it=doc.begin(); it!=doc.end(); ++it) {
//    string key, value;
//
//    if (it->second.IsScalar()) {
//      key = it->first.as<std::string>();
//      value = it->second.as<std::string>();
//      shared_ptr<void> temp(new string(value));
//      workflow_.Properties.insert({key, temp});
//    } else if (it->second.IsSequence()) {
//      UnitDescription unit;
//      unit.Name = it->second.Tag();
//      if (!WorkflowLoader::GetUnit(it->second, &unit)) {
//        fprintf(stderr, "Bad yaml parser: can't extract unit\n");
//
//        return false;
//      }
//      WorkflowLoader::workflow_.Units.push_back(unit);
//    } else if (it->second.IsMap()){
//      printf("Map: lenght: %ld\n",it->second.size());
//
//      YAML::Node temp = it->second;
//      for (int i = 0; i < temp.size(); ++i) {
//        switch (temp[i].Type()) {
//          case YAML::NodeType::Null:
//            printf("Null;\n");
//            break;
//          case YAML::NodeType::Undefined:
//            printf("Undefined;\n");
//            break;
//          case YAML::NodeType::Scalar:
//            printf("Scalar;\n");
//            break;
//          case YAML::NodeType::Sequence:
//            printf("Sequence;\n");
//            break;
//          case YAML::NodeType::Map:
//            printf("Map : lenght: %ld\n",temp.size());
//
//            break;
//        }
//      }
//      return false;
//    }
//  }
  return true;
}

bool WorkflowLoader::GetUnit(const YAML::Node& doc, UnitDescription* unit) {
//  for (auto it=doc.begin(); it!=doc.end(); ++it) {
//    string key, value;
//
//    if (it->second.IsScalar() ==  true) {
//      key = it->first.as<std::string>();
//      value = it->second.as<std::string>();
//
//      shared_ptr<void> temp(new string(value));
//
//      unit->Properties.insert({key, temp});
//
//      if (key.find(string("link_to_")) != string::npos) {
//        string new_key = key.substr(string("link_to_").size() );
//        string link_to_file = kWorkFolder + value;
//
//        // Open extracted files
//        ifstream fstream(link_to_file, std::ios::in|std::ios::binary|
//                         std::ios::ate);
//        // Calculate size of float array to read from file
//        int array_size = fstream.tellg();
//
//        // Read array
//        float* weight = new float[array_size];
//        fstream.read((char*)weight,  array_size* sizeof(float));
//
//        shared_ptr<void> temp2(new float*(weight));
//
//        unit->Properties.insert({new_key, temp2});
//        printf("Array size of %s: %ld\n", new_key.c_str(),
//               array_size/sizeof(float));
//
//        fstream.close();
//      }
//    } else {
//      printf("Can't read unit info\n");
//      return false;
//    }
//  }
//
  return true;
}

void WorkflowLoader::PrintWorkflowStructure() {
  // Print properties
  printf("\nWorkFlow properties:\n");

  for (auto x : workflow_.Properties) {
    printf("%s: %s\n", x.first.c_str(),
           static_cast<string*>(x.second.get())->c_str());
  }
  // Print units and their properties
  for (unsigned i = 0; i < workflow_.Units.size(); ++i) {
    printf("\nUnit name: %s\n", workflow_.Units.at(i).Name.c_str());

    for (auto y : workflow_.Units.at(i).Properties) {
      printf("%s: %s\n", y.first.c_str(),
             static_cast<string*>(y.second.get())->c_str());
    }
  }
}

bool WorkflowLoader::ExtractArchive(const string& filename,
    const string& folder) {
  static const size_t kBlockSize = 10240;

  auto destroy_read_archive = [](archive* ptr) {
    archive_read_close(ptr);
    archive_read_free(ptr);
  };
  auto input_archive = std::unique_ptr<archive, decltype(destroy_read_archive)>(
      archive_read_new(), destroy_read_archive);
  auto destroy_write_archive = [](archive* ptr) {
    archive_write_close(ptr);
    archive_write_free(ptr);
  };
  auto ext = std::unique_ptr<archive, decltype(destroy_write_archive)>(
      archive_write_disk_new(), destroy_write_archive);
  archive_entry *entry;
  int r;

  try {
    archive_read_support_filter_all(input_archive.get());
    archive_read_support_format_all(input_archive.get());
    archive_read_support_format_tar(input_archive.get());
    if ((r = archive_read_open_filename(input_archive.get(), filename.c_str(),
                                        kBlockSize))) {
      fprintf(stderr, "archive_read_open_filename() : %s\n",
             archive_error_string(input_archive.get()));
    }

    do {
      r = archive_read_next_header(input_archive.get(), &entry);
      if (r == ARCHIVE_EOF)
        break;
      if (r != ARCHIVE_OK) {
        fprintf(stderr, "archive_read_next_header() : %s\n",
               archive_error_string(input_archive.get()));
      }
      auto path = string(folder) + archive_entry_pathname(entry);
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
    } while(true);
    return true;
  } catch(const std::exception& e) {
    fprintf(stderr, "Can't open archive : %s", e.what());
    return false;
  }

  return false; // It shouldn't get here.
}

bool WorkflowLoader::RemoveDirectory(const string& path) {
  {
  auto dir = std::unique_ptr<DIR, decltype(&closedir)>(
      opendir(path.c_str()), closedir);
    if (!dir) {  // if pdir wasn't initialised correctly
      fprintf(stderr,
              "Can't open directory to delete, check path + permissions\n");
      return false;  // return false to say "we couldn't do it"
    }  // end if

    dirent *pent = NULL;
    unsigned char isFile =0x8;  // magic number to find files in folder
    while ((pent = readdir(dir.get()))) {
      // while there is still something in the directory to list
      if (pent->d_type == isFile) {
        const char *file_to_delete;
        file_to_delete = string(path + "/" + string(pent->d_name)).c_str();
        remove(file_to_delete);
      }
    }
  // finally, let's clean up
  }
  if (rmdir(path.c_str()) == 0) return true;  // delete the directory
  return false;
}

int WorkflowLoader::CopyData(const archive& ar, archive *aw) {
  int res;
  const void *buff;
  size_t size;
#if ARCHIVE_VERSION >= 3000000
  int64_t offset;
#else
  off_t offset;
#endif
  do {
   res = archive_read_data_block(const_cast<archive*>(&ar), &buff, &size, &offset);
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
