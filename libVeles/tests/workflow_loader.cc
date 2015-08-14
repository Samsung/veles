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

#include <unistd.h>
#include <string>
#include <gtest/gtest.h>
#include <veles/workflow_loader.h>


namespace veles {

class WorkflowLoaderTest
    : public ::testing::Test,
      protected DefaultLogger<WorkflowLoaderTest, Logger::COLOR_VIOLET> {
 public:

  WorkflowLoaderTest() {

  }
};

TEST_F(WorkflowLoaderTest, MnistWorkflow) {

}

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
