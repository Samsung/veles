/*! @file workflow.h
 *  @brief New file description.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef TESTS_WORKFLOW_H_
#define TESTS_WORKFLOW_H_

#include <vector>
#include <gtest/gtest.h>
#include "inc/veles/workflow.h"

/** @brief Isolated workflow test using Unit Mock
 */
class WorkflowTest : public ::testing::TestWithParam<
                              std::tuple<size_t, size_t, size_t, size_t>> {
 protected:
  static const size_t kCount;

  virtual void SetUp() override;

  /** @brief Get expected result of applying UnitMock's repeating algorithm
   *  to the input vector.
   *  @detail Example for configuration of sizes: 5 2 3 5:
   *  in -> [1 2 3 4 5] -> [1 2] (*2) -> [1 2 1] (*4) -> [1 2 1 1 2] (*8) -> out
   */
  void GetExpected(std::vector<size_t>* out);

  Veles::Workflow workflow_;
};

#endif  // TESTS_WORKFLOW_H_
