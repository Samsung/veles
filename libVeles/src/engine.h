/*! @file engine.h
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

#ifndef SRC_ENGINE_H_
#define SRC_ENGINE_H_

#include <functional>
#include <thread>
#include "inc/veles/logger.h"
#include "inc/veles/engine.h"

namespace veles {

namespace internal {

class ThreadPool;

class ThreadPoolEngine
    : public Engine,
      protected DefaultLogger<ThreadPoolEngine, Logger::COLOR_BLUE> {
 public:
  explicit ThreadPoolEngine(
      size_t threads_number = std::thread::hardware_concurrency());

  virtual void Schedule(const Callable& callable) override;

 private:
  std::shared_ptr<ThreadPool> pool_;
};

}  // namespace internal

}  // namespace veles
#endif  // SRC_ENGINE_H_
