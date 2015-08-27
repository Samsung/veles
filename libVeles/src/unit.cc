/*! @file unit.cc
 *  @brief Unit class implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
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


#include "inc/veles/unit.h"
#include <list>
#include <veles/logger.h>
#include "src/engine.h"

namespace veles {

Unit::Unit(const std::shared_ptr<internal::Engine>& engine)
    : engine_(engine), output_(nullptr), gate_(false) {
}

void Unit::LinkFrom(const std::shared_ptr<Unit>& parent) {
  for (auto& next : links_from_) {
    if (next == parent) {
      WRN("Unit %s is already linked with unit %s", typeid(*this).name(),
          typeid(*parent.get()).name());
      return;
    }
  }
  parent->links_from_.push_back(shared_from_this());
  links_to_.push_back(parent);
}

void Unit::Initialize() {
  Reset();
}

void Unit::Run() {
  if (!Ready()) {
    throw UnexpectedRunException(typeid(*this).name());
  }
  DBG("Run()");
  Execute();
  gate_ = true;
  if (!links_from_.size()) {
    engine_->Finish();
  } else {
    for (auto next : links_from_) {
      if (next->Ready()) {
        engine_->Schedule([next]() { next->Run(); });
      }
    }
  }
}

bool Unit::Ready() const noexcept {
  for (auto& unit : links_to_) {
    if (!unit.lock()->gate()) {
      return false;
    }
  }
  return true;
}

void Unit::Reset() {
  gate_ = false;
  for (auto& unit : links_from_) {
    unit->gate_ = false;
  }
}

const Unit* Unit::BreadthFirstWalk(
    const std::function<WalkDecision(const Unit* unit)>& payload,
    bool include_self) const {
  const Unit* unit = nullptr;
  std::list<const Unit*> fifo;
  fifo.push_back(this);
  while (fifo.size()) {
    unit = fifo.front();
    fifo.pop_front();
    if (unit != this || include_self) {
      auto res = payload(unit);
      if (!res) {
        break;
      }
      if (res == WalkDecision::kIgnore) {
        continue;
      }
    }
    for (auto& next : unit->Children()) {
      fifo.push_back(next.get());
    }
  }
  return unit;
}

const Unit* Unit::DepthFirstWalk(
    const std::function<WalkDecision(const Unit* unit)>& payload,
    bool include_self) const {
  const Unit* unit = nullptr;
  std::list<const Unit*> lifo;
  lifo.push_back(this);
  while (lifo.size()) {
    unit = lifo.back();
    lifo.pop_back();
    if (unit != this || include_self) {
      auto res = payload(unit);
      if (!res) {
        break;
      }
      if (res == WalkDecision::kIgnore) {
        continue;
      }
    }
    for (auto& next : unit->Children()) {
      lifo.push_back(next.get());
    }
  }
  return unit;
}

const Unit* Unit::ReversedBreadthFirstWalk(
    const std::function<WalkDecision(const Unit* unit)>& payload,
    bool include_self) const {
  const Unit* unit = nullptr;
  std::list<const Unit*> fifo;
  fifo.push_back(this);
  while (fifo.size()) {
    unit = fifo.front();
    fifo.pop_front();
    if (unit != this || include_self) {
      auto res = payload(unit);
      if (!res) {
        break;
      }
      if (res == WalkDecision::kIgnore) {
        continue;
      }
    }
    for (auto& next : unit->Parents()) {
      fifo.push_back(next.lock().get());
    }
  }
  return unit;
}

const Unit* Unit::ReversedDepthFirstWalk(
    const std::function<WalkDecision(const Unit* unit)>& payload,
    bool include_self) const {
  const Unit* unit = nullptr;
  std::list<const Unit*> lifo;
  lifo.push_back(this);
  while (lifo.size()) {
    unit = lifo.back();
    lifo.pop_back();
    if (unit != this || include_self) {
      auto res = payload(unit);
      if (!res) {
        break;
      }
      if (res == WalkDecision::kIgnore) {
        continue;
      }
    }
    for (auto& next : unit->Parents()) {
      lifo.push_back(next.lock().get());
    }
  }
  return unit;
}

}  // namespace veles
