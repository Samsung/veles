/*! @file numpy_array_loader.h
 *  @brief Implementation of NumpyArrayLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2015 Samsung R&D Institute Russia
 */

#include <src/numpy_array_loader.h>

namespace veles {

NumpyArrayLoader::NumpyArrayLoader() {
}

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
    while (data[0] != last) {
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

const std::unordered_map<std::string, std::string>
NumpyArrayLoader::Header::kTypesDict = {
    {"b", typeid(int8_t).name()},
    {"i8", typeid(int8_t).name()},
    {"i16", typeid(int16_t).name()},
    {"i32", typeid(int32_t).name()},
    {"i64", typeid(int64_t).name()},
    {"u8", typeid(uint8_t).name()},
    {"u16", typeid(uint16_t).name()},
    {"u32", typeid(uint32_t).name()},
    {"u64", typeid(uint64_t).name()},
    {"f2", "float16"},
    {"f4", typeid(float).name()},
    {"f8", typeid(double).name()},
};

const std::unordered_map<std::string, int>
NumpyArrayLoader::Header::kSizesDict = {
    {"b", 1},
    {"i8", 1},
    {"i16", 2},
    {"i32", 4},
    {"i64", 8},
    {"u8", 1},
    {"u16", 2},
    {"u32", 4},
    {"u64", 8},
    {"f2", 2},
    {"f4", 4},
    {"f8", 8},
};

NumpyArrayLoader::Header::Header() : fortran_order(false) {
  memset(shape.data(), 0, shape.size() * sizeof(shape[0]));
}


int NumpyArrayLoader::Header::DtypeSize() const {
  auto it = kSizesDict.find(dtype);
  if (it == kSizesDict.end()) {
    throw NumpyArrayLoadingFailedException(
        std::string("unsupported dtype ") + dtype);
  }
  return it->second;
}

bool NumpyArrayLoader::Header::DtypeIsLittleEndian() const {
  return dtype[0] == '<';
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
  DBGI(loader, "header: dtype %s, fortran: %s, shape: (%s)",
       dtype.c_str(), fortran_order? "true" : "false", shape_str);
}

}  // namespace veles
