/*! @file numpy_array_loader.cc
 *  @brief Numpy array loading tests.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2015 © Samsung R&D Institute Russia
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


#include <gtest/gtest.h>
#include <veles/logger.h>  // NOLINT(*)
#include "src/numpy_array_loader.h"


class NumpyArrayLoaderTest :
    public ::testing::TestWithParam<std::tuple<int, int>>,
    protected veles::DefaultLogger<
        NumpyArrayLoaderTest, veles::Logger::COLOR_VIOLET> {
};

void InitializeMatrix(float* matrix, int size) {
  for (int i = 0; i < size; i++) {
    matrix[i] = i;
  }
}

void Transpose(const float* src, float* dst, int rows, int cols) {
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      dst[x * rows + y] = src[y * cols + x];
    }
  }
}

TEST_P(NumpyArrayLoaderTest, Transpose) {
  int rows, cols;
  std::tie(rows, cols) = GetParam();
  INF("Testing %d rows x %d cols...", rows, cols);
  int size = rows * cols;
  float matrix[size];
  InitializeMatrix(matrix, size);
  float reference[size];
  Transpose(matrix, reference, rows, cols);
  veles::NumpyArrayLoader::TransposeInplace(matrix, rows, cols);
  ASSERT_EQ(std::memcmp(matrix, reference, sizeof(matrix)), 0);
}

INSTANTIATE_TEST_CASE_P(
    NumpyArrayLoaderTests, NumpyArrayLoaderTest,
        ::testing::Values(std::make_tuple(10, 15),
                          std::make_tuple(9, 9),
                          std::make_tuple(9, 7),
                          std::make_tuple(8, 8),
                          std::make_tuple(7, 9),
                          std::make_tuple(7, 7),
                          std::make_tuple(1, 2),
                          std::make_tuple(2, 1),
                          std::make_tuple(1, 1)));

#include "tests/google/src/gtest_main.cc"

