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

#include "inc/veles/workflow.h"
#include <cassert>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include "inc/veles/engine.h"
#include "inc/veles/unit.h"
#include "src/memory_optimizer.h"
#include <simd/arithmetic.h>

namespace veles {

Workflow::Workflow(const std::string& name, const std::string& checksum,
                   const std::shared_ptr<Unit>& head,
                   const std::shared_ptr<Engine>& engine)
    : name_(name), checksum_(checksum), head_(head), engine_(engine),
      engine_key_(engine->RegisterOnFinish(std::bind(&Workflow::Reset, this))),
      input_(nullptr) {
  head_->set_workflow(this);
}

 Workflow::~Workflow() noexcept {
   engine_->UnregisterOnFinish(engine_key_);
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

void Workflow::Initialize(const void* input) {
  auto problem = StateMemoryOptimizationProblem();
  size_t boilerplate_size = internal::MemoryOptimizer().Optimize(&problem);
  boilerplate_ = std::shared_ptr<uint8_t>(
      malloc_aligned<uint8_t>(boilerplate_size), std::free);
  assert(!(reinterpret_cast<intptr_t>(input) & 0xF) &&
         "input must be aligned to 16 bytes");
  input_ = input;
  for (auto& node : problem) {
    auto unit = const_cast<Unit*>(reinterpret_cast<const Unit*>(node.data));
    unit->set_output(boilerplate_.get() + node.position);
  }
  head_->BreadthFirstWalk([](const Unit* unit) {
    const_cast<Unit*>(unit)->Initialize();
    return true;
  });
}

void Workflow::Run() {
  sync_.reset(new std::condition_variable());
  sync_mutex_.reset(new std::mutex());
  engine_->Schedule([this]() { head_->Run(); });
}

void Workflow::Wait() {
  std::unique_lock<std::mutex> lock(*sync_mutex_);
  sync_->wait(lock);
}

void Workflow::Reset() {
  if (head_) {
    head_->Reset();
  }
  sync_->notify_all();
}

void* Workflow::output() const noexcept {
  return Tail()->output();
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
    if (out_size & 0x1F) {
      out_size = (out_size & ~0x1F) + 0x20;
    }
    node.value = out_size;
    node.data = unit;
    int time = 0;
    unit->ReversedBreadthFirstWalk(
        [&time](const Unit*) { time++; return true; }, false);
    node.time_start = time;
    std::unordered_set<const Unit*> visited;
    visited.insert(head_.get());
    time = 0;
    head_->BreadthFirstWalk(
        [&time, &visited, &unit](const Unit* other) -> WalkDecision {
      if (other == unit) {
        time++;
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
    node.time_finish = time + unit->Children().size();
    DBG("%s: %d -> %d\n", unit->Uuid().c_str(), node.time_start,
        node.time_finish);
    assert(node.time_finish > node.time_start);
  }
  return std::move(nodes);
}

}  // namespace veles

