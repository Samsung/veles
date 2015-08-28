/*! @file engine.cc
 *  @brief Class which is responsible for scheduling unit runs.
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

#include "src/engine.h"
#include "src/thread_pool.h"
#include "inc/veles/engine.h"

namespace veles {

void Engine::Finish() {
  assert(this && "Engine is nullptr");
  for (auto& cb : callbacks_) {
    cb.second();
  }
}

int Engine::RegisterOnFinish(const Callable& callback) noexcept {
  assert(this && "Engine is nullptr");
  callbacks_[counter_] = callback;
  return counter_++;
}

bool Engine::UnregisterOnFinish(int key) noexcept {
  auto it = callbacks_.find(key);
  if (it == callbacks_.end()) {
    return false;
  }
  callbacks_.erase(it);
  return true;
}

namespace internal {

ThreadPoolEngine::ThreadPoolEngine(size_t threads_number)
    : pool_(new ThreadPool(threads_number)) {
  DBG("Launched a thread pool with %d threads",
      static_cast<int>(threads_number));
}

void ThreadPoolEngine::Schedule(const Callable& callable) {
  pool_->enqueue(callable);
}

}  // namespace internal

std::shared_ptr<Engine> GetEngine() {
  return std::make_shared<internal::ThreadPoolEngine>(2);
}

}  // namespace veles
