/*! @file make_unique.h
 *  @brief Overseen std::make_unique implementation. When std gets that function, delete this.
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

#ifndef INC_VELES_MAKE_UNIQUE_H_
#define INC_VELES_MAKE_UNIQUE_H_

#include <memory>
#include <simd/memory.h>

namespace std {

template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<typename T, typename D>
std::unique_ptr<T, D> uniquify(T* ptr, D destructor) {
  return std::unique_ptr<T, D>(ptr, destructor);
}

template<typename T, typename D>
std::unique_ptr<T, D*> uniquify(T* ptr, D* destructor) {
  return std::unique_ptr<T, D*>(ptr, *destructor);
}

template<typename T>
std::unique_ptr<T[], decltype(&std::free)> unique_aligned(size_t length) {
  auto mem = malloc_aligned(length * sizeof(T));
  return std::unique_ptr<T[], decltype(&std::free)>(
      reinterpret_cast<T*>(mem), std::free);
}

}  // namespace std

#endif  // INC_VELES_MAKE_UNIQUE_H_
