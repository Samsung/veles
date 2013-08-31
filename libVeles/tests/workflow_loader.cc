/*! @file workflow_loader.cc
 *  @brief Source for tests for class WorkflowLoder.
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
#include <unistd.h>
#include <string>
#include <gtest/gtest.h>


using std::string;
using std::static_pointer_cast;

namespace Veles {
void IterateThroughYAML(const YAML::Node& node);
void PrintfNodeType(const YAML::Node& node, const string prepend);

class WorkflowLoaderTest: public ::testing::Test {
 public:
  WorkflowLoaderTest() {
    current_path = "";
    char  currentPath[FILENAME_MAX];

    if (!getcwd(currentPath, sizeof(currentPath))) {
      fprintf(stderr, "Can't locate current directory\n");
    } else {
      current_path = currentPath;
    }
    string temp_path1 = current_path + "/workflow_files/";
    string temp_path2 = current_path + "/tests/workflow_files/";
    // Check existence of archive
    if (access(temp_path1.c_str(), 0) != -1) {
      current_path = temp_path1;  // "/workflow_desc_files/"
    } else if (access(temp_path2.c_str(), 0) != -1) {
      current_path = temp_path2;  // "/tests/workflow_desc_files/"
    } else {
      current_path = "";  // Error
    }
  }

  void MainTest2() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    // Check everything
    WorkflowLoader test;

    string temp = current_path + "workflow.tar.gz";
    ASSERT_NO_THROW(test.Load(temp));
    EXPECT_EQ(size_t(2), test.workflow_desc_.Units.size());

    int first_unit = 0, second_unit = 1;
    if (test.workflow_desc_.Units.at(0).Name == string("All2AllTanh")) {
      first_unit = 0;
      second_unit = 1;
    } else if (test.workflow_desc_.Units.at(1).Name == string("All2AllTanh")) {
      first_unit = 1;
      second_unit = 0;
    } else {
      FAIL();
    }
//      link_to_output: unit1output.bin
    EXPECT_EQ(string("All2AllTanh"), test.workflow_desc_.Units.at(first_unit).Name);
    EXPECT_EQ(string("All2All"), test.workflow_desc_.Units.at(second_unit).Name);
    EXPECT_EQ(string("unit0bias.bin"),
                  *std::static_pointer_cast<string>(
                      test.workflow_desc_.Units.at(first_unit).Properties.at(
                          "link_to_bias")));
    EXPECT_EQ(string("unit1bias.bin"),
                  *std::static_pointer_cast<string>(
                      test.workflow_desc_.Units.at(second_unit).Properties.at(
                          "link_to_bias")));
    EXPECT_EQ(string("unit0output.bin"),
                  *std::static_pointer_cast<string>(
                      test.workflow_desc_.Units.at(first_unit).Properties.at(
                          "link_to_output")));
    EXPECT_EQ(string("unit1output.bin"),
                  *std::static_pointer_cast<string>(
                      test.workflow_desc_.Units.at(second_unit).Properties.at(
                          "link_to_output")));
  }

  void MainTest() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    WorkflowLoader test;
    string temp = current_path + "neural_network.tar.gz";
    ASSERT_NO_THROW(test.Load(temp));
    EXPECT_EQ(size_t(2), test.workflow_desc_.Units.size()) << "Not equal";
    EXPECT_EQ(size_t(4), test.workflow_desc_.Properties.size());
    EXPECT_EQ(string("2"),
              *std::static_pointer_cast<string>(test.workflow_desc_.Properties.at(
                  "layers_number")));
    EXPECT_EQ(string("0"),
              *std::static_pointer_cast<string>(test.workflow_desc_.Properties.at(
                  "neural_network_type")));
    EXPECT_EQ(string("Feedforward neural network"),
              *std::static_pointer_cast<string>(test.workflow_desc_.Properties.at(
                  "neural_network_type_desc")));
    EXPECT_EQ(string("SaverUnit2"),
              *std::static_pointer_cast<string>(test.workflow_desc_.Properties.at(
                  "service_info")));
    int first_layer = 0, second_layer = 1;
    if (test.workflow_desc_.Units.at(0).Name == string("layer 0")) {
      first_layer = 0;
      second_layer = 1;
    } else if (test.workflow_desc_.Units.at(1).Name == string("layer 0")) {
      first_layer = 1;
      second_layer = 0;
    } else {
      FAIL();
    }
    EXPECT_EQ(string("layer 0"), test.workflow_desc_.Units.at(first_layer).Name);
    EXPECT_EQ(string("layer 1"), test.workflow_desc_.Units.at(second_layer).Name);
    EXPECT_EQ(string("tanh"),
              *std::static_pointer_cast<string>(
                  test.workflow_desc_.Units.at(first_layer).Properties.at(
                      "activation_function_descr")));
    EXPECT_EQ(string("softmax"),
              *std::static_pointer_cast<string>(
                  test.workflow_desc_.Units.at(second_layer).Properties.at(
                      "activation_function_descr")));
    EXPECT_EQ(string("layer0bias.bin"),
              *std::static_pointer_cast<string>(
                  test.workflow_desc_.Units.at(first_layer).Properties.at(
                      "link_to_bias")));
    EXPECT_EQ(string("layer1bias.bin"),
              *std::static_pointer_cast<string>(
                  test.workflow_desc_.Units.at(second_layer).Properties.at(
                      "link_to_bias")));
  }

  void TestPropertiesTable() {
    PropertiesTable testTable;
    auto testShare = std::make_shared<string>("Test shared");
    testTable.insert({"testTable", testShare});

    EXPECT_EQ(string("Test shared"),
              *static_cast<string*>(testTable.at("testTable").get()));
  }

  void GetUnitTest() {
    YAML::Node node = YAML::Load("{Samsung: Cool}");

    UnitDescription testUnit;
    testUnit.Name = "Test Name";

    WorkflowLoader test;
    test.GetUnit(node, &testUnit);

    EXPECT_EQ("Test Name", testUnit.Name);
    EXPECT_EQ("Cool",
              *static_cast<string*>(testUnit.Properties.at("Samsung").get()));
  }

  void ComplexYamlTest1() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    string temp = current_path + "default.yaml";
    ASSERT_NO_THROW(test.GetWorkflow(temp));

    string result = test.PrintWorkflowStructure();

    EXPECT_NE(string::npos, result.find("service_info : SaverUnit2"));
    EXPECT_NE(string::npos, result.find("layers_number : 2"));
    EXPECT_NE(string::npos, result.find("neural_network_type : 0"));
    EXPECT_NE(string::npos, result.find("neural_network_type_desc : Feedforward neural network"));
    EXPECT_NE(string::npos, result.find("Unit name: layer 0"));
    EXPECT_NE(string::npos, result.find("activation_function_descr : tanh"));
    EXPECT_NE(string::npos, result.find("width : 784"));
    EXPECT_NE(string::npos, result.find("layer_number : 0"));
    EXPECT_NE(string::npos, result.find("height : 100"));
    EXPECT_NE(string::npos, result.find("activation_function : 0"));
    EXPECT_NE(string::npos, result.find("Unit name: layer 1"));
    EXPECT_NE(string::npos, result.find("layer_number : 1"));
    EXPECT_NE(string::npos, result.find("height : 10"));
    EXPECT_NE(string::npos, result.find("activation_function_descr : softmax"));
    EXPECT_NE(string::npos, result.find("width : 100"));
    EXPECT_NE(string::npos, result.find("activation_function : 1"));
  }

  void TestExtractArchive() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    string pathToArchive = current_path + "test_archive.tar.gz";
    char tempFolderName[40] = "/tmp/workflow_desc_files_tmpXXXXXX";
    char* tempFolderName2 = mkdtemp(tempFolderName);
    // Check existence of temporary folder
    if (tempFolderName2 == nullptr) {
      fprintf(stderr, "Can't create temporary folder");
      FAIL();
    }
    // Check existence of archive
    if (access(pathToArchive.c_str(), 0) == -1) {
      // Delete temporary folder
      rmdir(string(tempFolderName2).c_str());
      // Printf error + FAIL
      fprintf(stderr, "Path doesn't exist : %s\n", pathToArchive.c_str());
      FAIL();
    }

    // Try to extract archive that doesn't exist. Expected false.
    EXPECT_ANY_THROW(test.ExtractArchive("unexisting_archive.tar.gz",
                                     "some_new_folder/"));
    // Try to extract test archive. Expected true.
    EXPECT_NO_THROW(test.ExtractArchive(pathToArchive, tempFolderName2));
    // Delete temp folder
    rmdir(string(string(tempFolderName2) + string("/test_archive")).c_str());
    rmdir(string(tempFolderName2).c_str());
  }

  void TestRemoveDirectory() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    string pathToArchive = current_path + "remove_folder_testing.tar.gz";

    char tempFolderName[40] = "/tmp/workflow_desc_files_tmp2XXXXXX";
    char* tempFolderName2 = mkdtemp(tempFolderName);
    // Check existence of temporary folder
    if (tempFolderName2 == nullptr) {
      fprintf(stderr, "Can't create temporary folder");
      FAIL();
    }
    // Check existence of archive
    if (access(pathToArchive.c_str(), 0) == -1) {
      // Delete temporary folder
      rmdir(string(tempFolderName2).c_str());
      fprintf(stderr,
             "Path doesn't exist : %s\n", pathToArchive.c_str());
      FAIL();
    }
    // Check deleting of non existing folder
    string temp("/tmp/workflow_desc_files_non_existing_folder/");
    EXPECT_ANY_THROW(test.RemoveDirectory(temp));
    temp = tempFolderName2;
    // Extract folder with files with archive and remove this folder
    ASSERT_NO_THROW(test.ExtractArchive(pathToArchive, temp));

    EXPECT_NO_THROW(test.RemoveDirectory(tempFolderName2)) <<
          "Can't delete folder " << tempFolderName2;
  }

  WorkflowLoader test;
  struct statfs *buffer;
  string current_path;
};

void PrintfNodeType(const YAML::Node& node, const string prepend) {
  switch (node.Type()) {
    case YAML::NodeType::Map:
      fprintf(stderr, "%s is Map, size: %ld\n", prepend.c_str(), node.size());
      break;
    case YAML::NodeType::Null:
      fprintf(stderr, "%s is Null, size: %ld\n", prepend.c_str(), node.size());
      break;
    case YAML::NodeType::Scalar:
      fprintf(stderr, "%s is Scalar: %s\n", prepend.c_str(),
             node.as<string>().c_str());
      break;
    case YAML::NodeType::Sequence:
      fprintf(stderr, "%s is Sequence, size: %ld\n", prepend.c_str(),
              node.size());
      break;
    case YAML::NodeType::Undefined:
      fprintf(stderr, "%s is Undefined, size: %ld\n", prepend.c_str(),
              node.size());
      break;
  }
}

void IterateThroughYAML(const YAML::Node& node) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    switch (it->Type()) {
      case YAML::NodeType::Null:
        fprintf(stderr, "Null;\n");
        break;
      case YAML::NodeType::Undefined:
        fprintf(stderr, "Undefined;\n");
        break;
      case YAML::NodeType::Scalar:
        fprintf(stderr, "Scalar \n");
        break;
      case YAML::NodeType::Sequence:
        fprintf(stderr, "Sequence;\n");
        break;
      case YAML::NodeType::Map:
        fprintf(stderr, "Map \n");
        break;
    }
  }
}

TEST_F(WorkflowLoaderTest, ExtractArchive) {
  TestExtractArchive();
}

TEST_F(WorkflowLoaderTest, RemoveDirectory) {
  TestRemoveDirectory();
}

TEST_F(WorkflowLoaderTest, ComplexYamlTest1) {
  ComplexYamlTest1();
}

TEST_F(WorkflowLoaderTest, GetUnitTest) {
  GetUnitTest();
}

TEST_F(WorkflowLoaderTest, TestPropertiesTable) {
  TestPropertiesTable();
}

TEST_F(WorkflowLoaderTest, MainTest) {
  MainTest();
  MainTest2();
}

}  // namespace Veles

#include "tests/google/src/gtest_main.cc"
