/*! @file numpy_array_loader.h
 *  @brief Declaration of NumpyArrayLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2015 Samsung R&D Institute Russia
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

#ifndef NUMPY_ARRAY_LOADER_H_
#define NUMPY_ARRAY_LOADER_H_

#include <cassert>
#include <cstring>
#include <istream>
#include <unordered_map>
#include <veles/logger.h>  // NOLINT(*)
#include <veles/poison.h>  // NOLINT(*)
#include "src/shared_array.h"
#include "src/endian2.h"

class NumpyArrayLoaderTest_Transpose_Test;

namespace veles {

class NumpyArrayLoadingFailedException : public std::exception {
 public:
  NumpyArrayLoadingFailedException(const std::string& reason)
      : message_(std::string("Failed to load NumPy array: \"") + reason + ".") {
  }

  virtual const char* what() const noexcept {
    return message_.c_str();
  }

 private:
  std::string message_;
};

template <class T, int D>
struct NumpyArray {
  constexpr static int SHAPE_MAX = 8;
  static_assert(std::is_arithmetic<T>::value,
                "the element type must be either integer or a floating point "
                "number.");
  static_assert(D > 0 && D < SHAPE_MAX,
                "the number of dimensions must be between 1 and "
                "NumpyArray::SHAPE_MAX.");
  typedef T ElementType;

  std::array<uint32_t, D> shape;
  bool transposed;
  shared_array<T> data;
};

class NumpyArrayLoader : protected DefaultLogger<NumpyArrayLoader,
                                                 Logger::COLOR_YELLOW> {
 public:
  NumpyArrayLoader();

  template <class T, int D, bool transposed=false>
  NumpyArray<T, D> Load(std::istream* src);

 private:
  friend class ::NumpyArrayLoaderTest_Transpose_Test;
  struct Header {
    static const std::unordered_map<std::string, int> kSizesDict;
    static const std::unordered_map<std::string, std::string> kTypesDict;

    Header();

    std::string dtype;
    bool fortran_order;
    std::array<uint32_t, NumpyArray<int, 1>::SHAPE_MAX> shape;

    int DtypeSize() const;
    bool DtypeIsLittleEndian() const;
    int Dimensions() const;
    int64_t SizeInBytes() const;
    int SizeInElements() const;
    template <class T>
    bool DtypeIsTheSameAs() const;
    void DebugDescribe(NumpyArrayLoader* loader) const;
  };

  Header ParseHeader(char* data);
  template <class T>
  static void TransposeInplace(T* matrix, int rows, int columns);
};

template <class T, int D, bool transposed>
NumpyArray<T, D> NumpyArrayLoader::Load(std::istream* src) {
  static_assert(!transposed || D > 1, "No point in transposing 1D arrays");
  char signature[7];
  signature[6] = 0;
  src->read(signature, 6);
  if (src->eof()) {
    throw NumpyArrayLoadingFailedException("failed to read the signature");
  }
  if (reinterpret_cast<uint8_t>(signature[0]) != 0x93 ||
      strcmp(signature + 1, "NUMPY")) {
    throw NumpyArrayLoadingFailedException("signature is invalid");
  }
  uint8_t major, minor;
  if (src->eof()) {
    throw NumpyArrayLoadingFailedException("failed to read the format version");
  }
  (*src) >> major;
  if (src->eof()) {
    throw NumpyArrayLoadingFailedException("failed to read the format version");
  }
  (*src) >> minor;
  if (major > 1) {
    throw NumpyArrayLoadingFailedException(
        std::string("unsupported format version ") + std::to_string(major));
  }
  DBG("Loading format version %d.%d", major, minor);
  if (src->eof()) {
    throw NumpyArrayLoadingFailedException("failed to read the header size");
  }
  uint16_t header_size;
  (*src) >> header_size;
#ifdef BOOST_BIG_ENDIAN
  header_size = (header_size << 8) | (header_size >> 8)
#endif
  Header&& header;
  {
    std::unique_ptr<char[]> raw_header(new char[header_size + 1]);
    raw_header[header_size] = 0;
    src->read(raw_header.get(), header_size);
    if (src->eof()) {
      throw NumpyArrayLoadingFailedException("failed to read the header");
    }
    header = ParseHeader(raw_header.get());
  }
  header.DebugDescribe(this);
  if (header.Dimensions() != D) {
    throw NumpyArrayLoadingFailedException(
        std::string("array dimensions mismatch: requested ") +
        std::to_string(D) + ", read " + std::to_string(header.Dimensions()) +
        ")");
  }
#ifdef BOOST_BIG_ENDIAN
  if (header.DtypeIsLittleEndian()) {
    throw NumpyArrayLoadingFailedException(
        "endianness mismatch: native is big, package is little");
  }
#else
  if (!header.DtypeIsLittleEndian()) {
    throw NumpyArrayLoadingFailedException(
        "endianness mismatch: native is little, package is big");
  }
#endif
  auto data_size = header.SizeInBytes();
  std::unique_ptr<char[]> raw_data(new(std::nothrow) char[data_size]);
  if (!raw_data) {
    throw NumpyArrayLoadingFailedException(
        std::string("out of memory (") + std::to_string(data_size) +
        " bytes required)");
  }
  auto pos = src->tellg();
  src->read(raw_data.get(), data_size);
  if (src->tellg() - pos != data_size) {
    throw NumpyArrayLoadingFailedException("failed to read the array contents");
  }
  if (header.fortran_order != transposed) {
    // transpose
  }
  if (!header.DtypeIsTheSameAs<T>()) {
    std::unique_ptr<T[]> cdata(new T[header.SizeInElements()]);
  }

  NumpyArray<T, D> arr;
  arr.data = shared_array<T>(reinterpret_cast<T*>(raw_data.get()));
  raw_data.release();
  arr.transposed = transposed;
  for (int i = 0; i < D; i++) {
    arr.shape[i] = header.shape[i];
  }
  return std::move(arr);
}

template <class T>
void NumpyArrayLoader::TransposeInplace(T* matrix, int rows, int columns) {
  int size = rows * columns;
  assert(size > 0 && "Matrix size must be greater than 0");
  if (size == 1) {
    return;
  }
  if (rows == columns) {
    for (int y = 0; y < rows; y++) {
      for (int x = y + 1; x < rows; x++) {
        std::swap(matrix[y * rows + x], matrix[x * rows + y]);
      }
    }
    return;
  }
  int modulo = size - 1;
  int visited_size = size / 64 + ((size & (64 - 1)) > 0);
  std::unique_ptr<uint64_t[]> visited(new uint64_t[visited_size]);
  memset(visited.get(), 0xFF, visited_size * 8);
  visited[0]--;

  for (int i = 1; i < modulo;) {
    int cycle_first = i;
    T t = matrix[i];
    do {
      int next = (i * rows) % modulo;
      std::swap(matrix[next], t);
      visited[i / 64] -= 1ull << (i & (64 - 1));
      i = next;
    }
    while (i != cycle_first);

    for (i = 0; i < modulo && !visited[i / 64]; i += 64) {}
    i += __builtin_ctzll(visited[i / 64]);
  }
}

template <class T>
bool NumpyArrayLoader::Header::DtypeIsTheSameAs() const {
  return !strcmp(typeid(T).name(),
                 kTypesDict.find(dtype.c_str() + 1)->second.c_str());
}

}  // namespace veles
#endif  // NUMPY_ARRAY_LOADER_H_
