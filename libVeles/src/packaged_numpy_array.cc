/*! @file packaged_numpy_array.cc
 *  @brief PackagedNumpyArray implementation.
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

#include "inc/veles/packaged_numpy_array.h"
#include "src/main_file_loader.h"
#include "src/workflow_archive.h"

namespace veles {

PackagedNumpyArray::PackagedNumpyArray(
    const internal::NumpyArrayReference& ref,
    const std::shared_ptr<internal::WorkflowArchive>& war)
    : file_name_(ref.file_name()), war_(war) {
}

std::shared_ptr<std::istream> PackagedNumpyArray::GetStream() const {
  return war_->GetStream(file_name_);
}

}  // namespace veles
