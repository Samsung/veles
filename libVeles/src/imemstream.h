/*! @file imemstream.h
 *  @brief istream which reads from buffer in memory.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
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

#ifndef SRC_IMEMSTREAM_H_
#define SRC_IMEMSTREAM_H_

#include <istream>
#include "src/shared_array.h"

namespace veles {

template <class T>
class membuf: public std::streambuf {
  static_assert(sizeof(T) == 1, "Only 1-byte types are accepted");

 public:
  explicit membuf(const shared_array<T>& mem) : mem_(mem) {
    char* p(const_cast<char*>(reinterpret_cast<const char*>(mem.get_raw())));
    this->setg(p, p, p + mem.size());
  }

  membuf(const std::unique_ptr<T[]>& mem, size_t size) {
    char* p(const_cast<char*>(reinterpret_cast<const char*>(mem.get())));
    this->setg(p, p, p + size);
  }

  membuf(const T* mem, size_t size) {
    char* p(const_cast<char*>(reinterpret_cast<const char*>(mem)));
    this->setg(p, p, p + size);
  }

 protected:
  virtual pos_type seekoff(off_type offset, std::ios::seekdir seekdir,
      std::ios::openmode = std::ios::in | std::ios::out) override {
    switch (seekdir) {
      case std::ios::beg:
        assert(eback() + offset <= egptr());
        _M_in_cur = eback() + offset;
        break;
      case std::ios::cur:
        assert(gptr() + offset <= egptr());
        _M_in_cur += offset;
        break;
      case std::ios::end:
        assert(egptr() - offset >= eback());
        _M_in_cur = egptr() - offset;
        break;
      default:
        break;
    }
    return pos_type(off_type(gptr() - eback()));
  }

 private:
  shared_array<char> mem_;
};

template <class T>
struct imemstream: public virtual membuf<T>, public std::istream {
  static_assert(sizeof(T) == 1, "Only 1-byte types are accepted");

  explicit imemstream(const shared_array<T>& mem)
    : membuf<T>(mem),
      std::istream(static_cast<std::streambuf*>(this)) {
  }

  imemstream(const std::unique_ptr<T[]>& mem, size_t size)
    : membuf<T>(mem, size),
      std::istream(static_cast<std::streambuf*>(this)) {
  }

  imemstream(const T* mem, size_t size)
    : membuf<T>(mem, size),
      std::istream(static_cast<std::streambuf*>(this)) {
  }
};

}  // namespace veles

#endif  // SRC_IMEMSTREAM_H_
