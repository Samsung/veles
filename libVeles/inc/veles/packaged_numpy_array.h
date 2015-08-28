/*! @file packaged_numpy_array.h
 *  @brief PackagedNumpyArray class definition. This class is a proxy between
 *  NumpyArrayReference and NumpyArray.
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

#ifndef INC_VELES_PACKAGED_NUMPY_ARRAY_H_
#define INC_VELES_PACKAGED_NUMPY_ARRAY_H_

#include <veles/numpy_array_loader.h>

namespace veles {

namespace internal {

class NumpyArrayReference;
class WorkflowArchive;

}  // namespace internal

class PackagedNumpyArray {
 public:
  PackagedNumpyArray(const internal::NumpyArrayReference& ref,
                     const std::shared_ptr<internal::WorkflowArchive>& war);

  template <class T, int D, bool transposed=false>
  NumpyArray<T, D> get() const {
    auto stream = GetStream();
    return loader_.Load<T, D, transposed>(stream.get());
  }

 private:
  std::shared_ptr<std::istream> GetStream() const;

  const std::string file_name_;
  std::shared_ptr<internal::WorkflowArchive> war_;
  const internal::NumpyArrayLoader loader_;
};

}  // namespace veles

#endif  // INC_VELES_PACKAGED_NUMPY_ARRAY_H_
