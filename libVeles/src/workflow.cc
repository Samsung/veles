/*! @file workflow.cc
 *  @brief VELES workflow
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

#include <cassert>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include "inc/veles/workflow.h"
#include "src/memory_optimizer.h"

namespace veles {

Workflow::Workflow(const std::string& name, const std::string& checksum,
                   const std::shared_ptr<Unit>& head,
                   const std::shared_ptr<Engine>& engine)
    : name_(name), checksum_(checksum), head_(head), engine_(engine) {
}

void Workflow::Clear() noexcept {
  head_ = nullptr;
}

size_t Workflow::Size() const noexcept {
  size_t count = 0;
  head_->BreadthFirstWalk([&count](const Unit*) { return ++count; });
  return count;
}

std::shared_ptr<Unit> Workflow::Tail() const noexcept {
  return const_cast<Unit*>(head_->DepthFirstWalk([](const Unit* ptr) {
    return bool(ptr->Children().size());
  }))->shared_from_this();
}

void* Workflow::malloc_aligned_void(size_t size) {
#ifndef __ANDROID__
  void *ptr;
  return posix_memalign(&ptr, 64, size) == 0 ? ptr : nullptr;
#else
  return reinterpret_cast<float*>(memalign(64, size));
#endif
}

void Workflow::Initialize(const void* input) {
  auto problem = StateMemoryOptimizationProblem();
  size_t boilerplate_size = internal::MemoryOptimizer().Optimize(&problem);
  boilerplate_ = std::shared_ptr<uint8_t>(
      malloc_aligned<uint8_t>(boilerplate_size), std::free);
  assert(!(reinterpret_cast<intptr_t>(input) & 0xF) &&
         "input must be aligned to 16 bytes");
  for (auto& node : problem) {
    auto unit = const_cast<Unit*>(reinterpret_cast<const Unit*>(node.data));
    unit->set_output(boilerplate_.get() + node.position);
  }
}

void Workflow::Run() {
  head_->Run();
}

std::vector<internal::MemoryNode>
Workflow::StateMemoryOptimizationProblem() const {
  std::list<const Unit*> units;
  head_->BreadthFirstWalk([&units](const Unit* unit) {
    units.push_back(unit);
    return true;
  });
  std::vector<internal::MemoryNode> nodes(units.size());
  int i = 0;
  for (auto unit : units) {
    auto& node = nodes[i++];
    auto out_size = unit->OutputSize();
    if (out_size & 0xF) {
      out_size = (out_size & ~0xF) + 0x10;
    }
    node.value = out_size;
    node.data = unit;
    int time = 0;
    unit->ReversedBreadthFirstWalk([&time](const Unit*) { return ++time; });
    node.time_start = time;
    std::unordered_set<const Unit*> visited;
    visited.insert(head_.get());
    time = 0;
    head_->BreadthFirstWalk(
        [&time, &visited, &unit](const Unit* other) -> WalkDecision {
      if (other == unit) {
        return WalkDecision::kIgnore;
      }
      for (auto& parent : other->Parents()) {
        if (visited.find(parent.lock().get()) == visited.end()) {
          return true;
        }
      }
      visited.insert(other);
      time++;
      return true;
    });
    node.time_finish = time;
  }
  return std::move(nodes);
}

}  // namespace veles

