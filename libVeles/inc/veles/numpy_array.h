/*! @file numpy_array.h
 *  @brief Numpy array related classes.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2015 Samsung R&D Institute Russia
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

#ifndef INC_VELES_NUMPY_ARRAY_H_
#define INC_VELES_NUMPY_ARRAY_H_

#include <array>
#include <istream>
#include <type_traits>
#include <veles/shared_array.h>
#include <veles/numpy_array_loader.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

struct NumpyArrayBase {
  constexpr static int SHAPE_MAX = 8;
};

template <class T, int D>
struct NumpyArray : NumpyArrayBase {
  static_assert(std::is_arithmetic<T>::value,
                "the element type must be either integer or a floating point "
                "number.");
  static_assert(D > 0 && D < SHAPE_MAX,
                "the number of dimensions must be between 1 and "
                "NumpyArray::SHAPE_MAX.");
  typedef T ElementType;

  std::array<uint32_t, D> shape;
  bool transposed;
  shared_array<T> data;
};

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

}  // namespace veles

#endif  // INC_VELES_NUMPY_ARRAY_H_
