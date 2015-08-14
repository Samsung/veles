/*! @file workflow.h
 *  @brief New file description.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
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

#ifndef TESTS_WORKFLOW_H_
#define TESTS_WORKFLOW_H_

#include <vector>
#include <gtest/gtest.h>
#include "inc/veles/workflow.h"

/** @brief Isolated workflow test using Unit Mock
 */
class WorkflowTest :
    public ::testing::TestWithParam<
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

  veles::Workflow workflow_;
};

#endif  // TESTS_WORKFLOW_H_
