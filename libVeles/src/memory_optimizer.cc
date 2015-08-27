/*! @file memory_optimizer.cc
 *  @brief Class to pack sliding blocks to reach the minimal height.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *@section Copyright
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

#include "src/memory_optimizer.h"
#include <algorithm>
#include <utility>

namespace veles {

namespace internal {

size_t MemoryOptimizer::Optimize(std::vector<MemoryNode>* nodes) const {
  // Determine the overall time
  int overall_time = 0;
  for (auto& node : *nodes) {
    if (node.time_finish > overall_time) {
      overall_time = node.time_finish;
    }
  }
  size_t max_height = 0;
  map solution(overall_time);
  // Simple greedy algorithm
  std::sort(nodes->begin(), nodes->end(),
            [](const MemoryNode& a, const MemoryNode& b) {
    return a.value > b.value;
  });
  for (auto& node : *nodes) {
    size_t pos = FindLowestPosition(solution, node);
    for (int t = node.time_start; t < node.time_finish; t++) {
      auto& column = solution[t];
      int ip = 0;
      for (auto& pair : column) {
        if (pair.second <= pos) {
          ip++;
        } else if (pair.first > pos) {
          break;
        }
      }
      column.emplace(std::next(column.begin(), ip), pos, pos + node.value);
    }
    node.position = pos;
    size_t top = pos + node.value;
    if (top > max_height) {
      max_height = top;
    }
  }
  return max_height;
}

size_t MemoryOptimizer::FindLowestPosition(
    const map& map, const MemoryNode& node) {
  size_t pos = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int t = node.time_start; t < node.time_finish; t++) {
      auto& column = map[t];
      size_t top = pos + node.value;
      if (column.back().second <= pos) {
        continue;
      }
      for (auto& pair : column) {
        if (pair.first < top && pair.second > pos) {
          pos = pair.second;
          top = pos + node.value;
          changed = true;
        }
      }
    }
  }
  return pos;
}

void MemoryOptimizer::Print(const std::vector<MemoryNode>& nodes,
                            std::ostream* out) const {
  for (auto& node : nodes) {
    (*out) << node.position << '\t' << node.value << '\t' << node.time_start <<
        '\t' << node.time_finish << std::endl;
  }
}

}  // namespace internal

}  // namespace veles
