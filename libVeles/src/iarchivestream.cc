/*! @file iarchivestream.cc
 *  @brief istream which reads from libarchive's struct archive *.
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

#include "src/iarchivestream.h"
#include <libarchive/libarchive/archive.h>  // NOLINT(*)

archbuf::archbuf(const std::shared_ptr<archive>& archive)
    : archive_(archive), read_(0) {
  auto end = buffer + sizeof(buffer);
  setg(end, end, end);
}

archbuf::int_type archbuf::underflow () {
  assert(gptr() == egptr());
  auto read = archive_read_data(archive_.get(), buffer, sizeof(buffer));
  if (read == 0) {
    return EOF;
  }
  read_ += read;
  setg(buffer, buffer, buffer + read);
  return traits_type::to_int_type(*gptr());
}

archbuf::pos_type archbuf::seekoff(
    off_type offset, std::ios::seekdir seekdir, std::ios::openmode) {
  switch (seekdir) {
    case std::ios::beg:
      offset -= archive_buffer_pos();
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
  return read_ - pos_type(off_type(egptr() - gptr()));
}

size_t archbuf::archive_buffer_pos() const noexcept {
  return read_ - (egptr() - eback());
}
