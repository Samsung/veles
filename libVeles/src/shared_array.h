/*!
 * Copyright (c) 2014, Samsung Electronics Co.,Ltd.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of Samsung Electronics Co.,Ltd..
 *
 */

/*! @file shared_array.h
 *  @brief std::shared_ptr for arrays.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2014 Samsung R&D Institute Russia
 */

#ifndef SRC_SHARED_ARRAY_H_
#define SRC_SHARED_ARRAY_H_

#include <algorithm>
#include <memory>

namespace veles {

/** @brief Class to provide shared array storage. */
template<typename T>
class shared_array {
 public:
  typedef T element_type;

  shared_array();
  /** @brief Creates shared array with corresponding size. */
  explicit shared_array(size_t size);
  /** @brief Creates shared array based on raw pointer and data size. */
  shared_array(T* ptr, size_t size);
  /** @brief Creates shared array based on shared pointer and data size. */
  shared_array(const std::shared_ptr<T>& sp, size_t size);

  shared_array(const shared_array<T>& array);
  shared_array<T>& operator=(const shared_array<T>& array);

  /** @brief Swaps data of this array with array in argument. */
  inline void swap(shared_array<T>& array);

  /** @brief Resets data of shared array. */
  inline void reset();

  /*!
   * @brief Resets data of shared array to corresponding shared pointer and
   * data size.
   */

  inline void reset(const std::shared_ptr<T>& sp, size_t size);
  /*!
   * @brief Resets data of shared array to corresponding raw pointer and
   * data size.
   */
  inline void reset(T* ptr, size_t size);

  const T& operator[](std::ptrdiff_t i) const noexcept;
  T& operator[](std::ptrdiff_t i) noexcept;

  /** @brief Checks for the existence of data. */
  operator bool() const noexcept;
  /** @brief Convert data pointer to const data pointer. */
  operator shared_array<const T>() const;

  /** @brief Returns shared pointer to array data. */
  std::shared_ptr<T> get() const noexcept { return sp_; }
  /** @brief Returns raw pointer to array data. */
  T* get_raw() const noexcept { return sp_.get(); }
  /** @brief Returns true if this shared array data is unique. */
  bool unique() const noexcept { return sp_.unique(); }
  /** @brief Returns number of references on shared array data. */
  int use_count() const noexcept { return sp_.use_count(); }
  /** @brief Returns number of elements in shared array. */
  size_t size() const noexcept { return size_; }
  /** @brief Returns memory size occupied by shared array. */
  size_t memsize() const noexcept { return sizeof(T) * size_; }

 private:
  std::shared_ptr<T> sp_;  // shared memory, which contains host array
  size_t size_;            // number of elements in array
};

template <typename T>
shared_array<T>::shared_array() : size_(0) {
}

template <typename T>
shared_array<T>::shared_array(size_t size)
    : sp_(new T[size], std::default_delete<T[]>()),
      size_(size) {
}

template <typename T>
shared_array<T>::shared_array(T* ptr, size_t size)
    : sp_(ptr, std::default_delete<T[]>()),
      size_(size) {
}

template <typename T>
shared_array<T>::shared_array(const std::shared_ptr<T>& sp, size_t size)
    : sp_(sp),
      size_(size) {
}

template <typename T>
shared_array<T>::shared_array(const shared_array<T>& array)
    : sp_(array.sp_),
      size_(array.size_) {
}

template <typename T>
shared_array<T>& shared_array<T>::operator=(const shared_array<T>& array) {
  if (this != &array) {
    sp_ = array.sp_;
    size_ = array.size_;
  }
  return *this;
}

template <typename T>
inline void shared_array<T>::reset() {
  sp_.reset();
  size_ = 0;
}

template <typename T>
inline void shared_array<T>::reset(const std::shared_ptr<T>& sp, size_t size) {
  sp_ = sp;
  size_ = size;
}

template <typename T>
inline void shared_array<T>::reset(T* ptr, size_t size) {
  sp_.reset(ptr, std::default_delete<T[]>());
  size_ = size;
}

template <typename T>
inline void shared_array<T>::swap(shared_array<T>& array) {
  sp_.swap(array.sp_);
  std::swap(size_, array.size_);
}

template <typename T>
const T& shared_array<T>::operator[](std::ptrdiff_t i) const noexcept {
  return *(sp_.get() + i);
}

template <typename T>
T& shared_array<T>::operator[](std::ptrdiff_t i) noexcept {
  return *(sp_.get() + i);
}

template <typename T>
shared_array<T>::operator bool() const noexcept {
  return sp_.get() != nullptr;
}

template <typename T>
shared_array<T>::operator shared_array<const T>() const {
  return shared_array<const T>(sp_, size_);
}

template<class T>
bool operator==(const shared_array<T>& a,
                const shared_array<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template<class T>
bool operator!=(const shared_array<T>& a, const shared_array<T>& b) {
  return !(a == b);
}

template<class T>
bool operator<(const shared_array<T>& a, const shared_array<T>& b) {
  if (a.size() < b.size()) return true;
  else if (a.size() > b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] < b[i]) return true;
    else if (a[i] > b[i]) return false;
  }
  return false;
}

template<class T>
void swap(shared_array<T>& a, shared_array<T>& b) {
  a.swap(b);
}

}  // namespace veles

#endif  // SRC_SHARED_ARRAY_H_
