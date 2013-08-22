/*! @file make_unique.h
 *  @brief Overseen std::make_unique implementation. When std gets that function, delete this.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_VELES_MAKE_UNIQUE_H_
#define INC_VELES_MAKE_UNIQUE_H_

#include <memory>

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

}  // namespace std

#endif  // INC_VELES_MAKE_UNIQUE_H_
