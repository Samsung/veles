/*! @file imemstream.h
 *  @brief New file description.
 *  @author markhor
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef IMEMSTREAM_H_
#define IMEMSTREAM_H_

#include <istream>

struct membuf: std::streambuf {
  membuf(char const* base, size_t size) {
    char* p(const_cast<char*>(base));
    this->setg(p, p, p + size);
  }

  virtual pos_type seekoff(off_type offset, std::ios::seekdir seekdir,
      std::ios::openmode = std::ios::in | std::ios::out) override {
    switch (seekdir) {
      case std::ios::beg:
        _M_in_cur = _M_in_beg + offset;
        break;
      case std::ios::cur:
        _M_in_cur += offset;
        break;
      case std::ios::end:
        _M_in_cur = _M_in_end;
        break;
      default:
        break;
    }
    return pos_type(off_type(_M_in_cur - _M_in_beg));
  }
};

struct imemstream: virtual membuf, public std::istream {
  imemstream(char const* base, size_t size)
    : membuf(base, size)
    , std::istream(static_cast<std::streambuf*>(this)) {
  }
};

#endif  // IMEMSTREAM_H_
