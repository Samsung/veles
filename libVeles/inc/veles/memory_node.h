/*! @file memory_node.h
 *  @brief MemoryNode struct definition.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
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

#ifndef INC_VELES_MEMORY_NODE_H_
#define INC_VELES_MEMORY_NODE_H_

#include <cstddef>

namespace veles {

namespace internal {

struct MemoryNode {
  MemoryNode() : time_start(-1), time_finish(-1), value(-1), position(-1),
                 data(nullptr) {
  }

  int time_start;
  int time_finish;
  size_t value;
  size_t position;
  const void* data;
};

}  // namespace internal

}  // namespace veles

#endif  // INC_VELES_MEMORY_NODE_H_
