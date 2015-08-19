/*! @file numpy_array_loader.h
 *  @brief Implementation of NumpyArrayLoader class.
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

#include "src/numpy_array_loader.h"
#include <simd/arithmetic.h>

namespace veles {

NumpyArrayLoader::Header NumpyArrayLoader::ParseHeader(char* data) {
  if (data[0] != '{') {
    throw NumpyArrayLoadingFailedException("failed to parse the header");
  }
  data++;

  auto parse_token = [&]() -> char* {
    char first;
    do {
      first = data++[0];
    } while (first && (first == ':' || first == ' '));
    if (!first) {
      return nullptr;
    }
    char last;
    char* start;
    bool quote = first == '\'';
    if (quote || first == '(') {
      last = quote? '\'' : ')';
      start = data;
    } else {
      last = ',';
      start = data - 1;
    }
    while (data[0] && data[0] != last) {
      data++;
    }
    data++[0] = 0;
    if (last != ',') {
      data++;
    }
    return start;
  };

  Header header;
  for (int i = 0; i < 3; i++) {
    auto key = parse_token();
    auto value = parse_token();
    if (!strcmp(key, "descr")) {
      header.dtype = value;
    } else if (!strcmp(key, "fortran_order")) {
      header.fortran_order = strcmp(value, "False");
    } else if (!strcmp(key, "shape")) {
      const char* token;
      data = value;
      int i = 0;
      do {
        token = parse_token();
      } while (token && (header.shape[i++] = std::atoi(token)));
    }
  }
  return header;
}

void NumpyArrayLoader::TransposeInplace(int rows, int columns, int esize,
                                        char* matrix) {
  int size = rows * columns;
  assert(size > 0 && "Matrix size must be greater than 0");
  if (size == 1) {
    return;
  }
  if (rows == columns) {
    for (int y = 0; y < rows; y++) {
      for (int x = y + 1; x < rows; x++) {
        std::swap_ranges(matrix + (y * rows + x) * esize,
                         matrix + (y * rows + x + 1) * esize,
                         matrix + (x * rows + y) * esize);
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
    char t[esize];
    memcpy(t, matrix + i * esize, esize);
    do {
      int next = (i * rows) % modulo;
      std::swap_ranges(matrix + next * esize, matrix + (next + 1) * esize, t);
      visited[i / 64] -= 1ull << (i & (64 - 1));
      i = next;
    }
    while (i != cycle_first);

    for (i = 0; i < modulo && !visited[i / 64]; i += 64) {}
    i += __builtin_ctzll(visited[i / 64]);
  }
}

void NumpyArrayLoader::ConvertTypeF16(const uint16_t* src, int size,
                                      float* dst) {
  float16_to_float(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeF16(const uint16_t* src, int size,
                                      uint16_t* dst) {
  ConvertTypeSame(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeI16(const int16_t* src, int size,
                                      float* dst) {
  int16_to_float(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeI16(const int16_t* src, int size,
                                      int16_t* dst) {
  ConvertTypeSame(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeI16(const int16_t* , int, uint16_t*) {
  assert(false && "Unreachable");
}

void NumpyArrayLoader::ConvertTypeI16(const int16_t*, int, int32_t*) {
  assert(false && "Unreachable");
}

void NumpyArrayLoader::ConvertTypeI32(const int32_t* src, int size,
                                      float* dst) {
  int32_to_float(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeI32(const int32_t* src, int size,
                                      int32_t* dst) {
  ConvertTypeSame(src, size, dst);
}

void NumpyArrayLoader::ConvertTypeI32(const int32_t*, int, uint16_t*) {
  assert(false && "Unreachable");
}

void NumpyArrayLoader::ConvertTypeI32(const int32_t*, int, int16_t*) {
  assert(false && "Unreachable");
}

const std::unordered_map<std::string, std::string>
NumpyArrayLoader::Header::kTypesDict = {
    {"b1", typeid(int8_t).name()},
    {"i1", typeid(int8_t).name()},
    {"i2", typeid(int16_t).name()},
    {"i4", typeid(int32_t).name()},
    {"i8", typeid(int64_t).name()},
    {"u1", typeid(uint8_t).name()},
    {"u2", typeid(uint16_t).name()},
    {"u4", typeid(uint32_t).name()},
    {"u8", typeid(uint64_t).name()},
    {"f2", "float16"},
    {"f4", typeid(float).name()},
    {"f8", typeid(double).name()},
};

NumpyArrayLoader::Header::Header() : fortran_order(false) {
  memset(shape.data(), 0, shape.size() * sizeof(shape[0]));
}


int NumpyArrayLoader::Header::DtypeSize() const {
  return std::stoi(dtype.substr(2));
}

bool NumpyArrayLoader::Header::DtypeIsLittleEndian() const {
  return dtype[0] == '<' || dtype[0] == '|';
}

int NumpyArrayLoader::Header::Dimensions() const {
  int i = 0;
  while (i < static_cast<int>(shape.size()) && shape[i]) {
    i++;
  }
  return i;
}

int64_t NumpyArrayLoader::Header::SizeInBytes() const {
  return static_cast<int64_t>(SizeInElements()) * DtypeSize();
}

int NumpyArrayLoader::Header::SizeInElements() const {
  int prod = 1;
  int dims = Dimensions();
  for (int i = 0; i < dims; i++) {
    prod *= shape[i];
  }
  return prod;
}

void NumpyArrayLoader::Header::DebugDescribe(NumpyArrayLoader* loader) const {
  int dims = Dimensions();
  char shape_str[(5 + 2) * dims];
  char* ptr = shape_str;
  for (int i = 0; i < dims; i++) {
    if (i < dims - 1) {
      sprintf(ptr, "%d, ", shape[i]);
    } else {
      sprintf(ptr, "%d", shape[i]);
    }
    ptr += strlen(ptr);
  }
  DBGI(loader, "Header: dtype %s, fortran: %s, shape: (%s)",
       dtype.c_str(), fortran_order? "true" : "false", shape_str);
}

}  // namespace veles
