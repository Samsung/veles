/*! @file iarchivestream.h
 *  @brief istream which reads from libarchive's struct archive *.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
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

#ifndef SRC_IARCHIVESTREAM_H_
#define SRC_IARCHIVESTREAM_H_

#include <cassert>
#include <istream>

struct archive;

class archbuf: public std::streambuf {
 public:
  archbuf(archive* archive);

  virtual int_type underflow () override;

  virtual pos_type seekoff(off_type offset, std::ios::seekdir seekdir,
      std::ios::openmode = std::ios::in | std::ios::out) override;

  size_t archive_buffer_pos() const noexcept;

 protected:
  static constexpr int kBufferSize = UINT16_MAX + 1;

  archive* archive_;
  size_t read_;
  char buffer[kBufferSize];
};

struct iarchivestream: public virtual archbuf, public std::istream {
  iarchivestream(archive* archive)
    : archbuf(archive)
    , std::istream(static_cast<std::streambuf*>(this)) {
  }
};

#endif  // SRC_IARCHIVESTREAM_H_
