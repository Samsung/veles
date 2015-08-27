/*! @file memory_optimizer.h
 *  @brief Class to pack sliding blocks to reach the minimal height.
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

#ifndef SRC_MEMORY_OPTIMIZER_H_
#define SRC_MEMORY_OPTIMIZER_H_

#include <list>
#include <vector>
#include "inc/veles/memory_node.h"

namespace veles {

namespace internal {

class MemoryOptimizer {
 public:
  size_t Optimize(std::vector<MemoryNode>* nodes) const;

 private:
  typedef std::vector<std::list<std::pair<size_t, size_t>>> map;

  static size_t FindLowestPosition(const map& map, const MemoryNode& node);
};

}  // namespace internal

}  // namespace veles
#endif  // SRC_MEMORY_OPTIMIZER_H_
