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

#include <gtest/gtest.h>
#include <string>
#include "inc/veles/workflow_loader.h"

namespace Veles {

class WorkflowLoaderTest {
 public:
  WorkflowLoader test;

  bool YamlTest() {
    std::string temp("some string");
    return test.GetWorkflow(temp);
  }
};
} // namespace Veles


TEST(WorkflowLoader, DummyYamlTest) {
  Veles::WorkflowLoaderTest dummy;
  ASSERT_EQ(true, dummy.YamlTest());
}

#include "tests/google/src/gtest_main.cc"
