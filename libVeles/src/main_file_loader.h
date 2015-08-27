/*! @file main_file_loader.h
 *  @brief Declaration of MainFileLoader class.
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

#ifndef MAIN_FILE_LOADER_H_
#define MAIN_FILE_LOADER_H_

#include <istream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <variant/variant.hpp>
#include <veles/logger.h>  // NOLINT(*)

template<typename... Types>
using variant = mapbox::util::variant<Types...>;

namespace veles {

namespace internal {

class NumpyArrayReference {
 public:
  explicit NumpyArrayReference(const std::string& file_name)
      : file_name_(file_name) {}

  const std::string& file_name() const noexcept { return file_name_; }

 private:
  std::string file_name_;
};

using Property = variant<bool, int, float, std::string, NumpyArrayReference>;

class UnitDefinition {
 public:
  UnitDefinition(const std::string& name, const std::string& uuid);
  virtual ~UnitDefinition() = default;

  std::string name() const noexcept { return name_; }
  const uint8_t* uuid() const noexcept { return uuid_; }
  std::string uuid_str() const noexcept;
  const std::vector<std::shared_ptr<UnitDefinition>>&
  links() const noexcept {
    return links_;
  }
  template <class T>
  const T& get(const std::string& key) const;
  template <class T>
  T& get(const std::string& key);
  template <class T>
  void set(const std::string& key, const T& value);
  std::vector<std::string> PropertyNames() const noexcept;
  void Link(std::shared_ptr<UnitDefinition> def);

 private:
  std::string name_;
  uint8_t uuid_[16];
  std::vector<std::shared_ptr<UnitDefinition>> links_;
  std::unordered_map<std::string, Property> props_;
};

class WorkflowDefinition {
 public:
  WorkflowDefinition(const std::string& checksum, const std::string& name,
                     std::shared_ptr<UnitDefinition> start);
  virtual ~WorkflowDefinition() = default;
  std::string checksum() const noexcept { return checksum_; }
  std::string name() const noexcept { return name_; }
  std::shared_ptr<UnitDefinition> start() const noexcept { return start_; }

 private:
  std::string checksum_;
  std::string name_;
  std::shared_ptr<UnitDefinition> start_;
};

/// Reads and parses the main file with description of the package (e.g.,
/// "contents.json").
class MainFileLoader : protected DefaultLogger<MainFileLoader,
                                               Logger::COLOR_YELLOW> {
 public:
  virtual ~MainFileLoader() = default;

  WorkflowDefinition Load(std::istream* src);
};

template <class T>
const T& UnitDefinition::get(const std::string& key) const {
  auto val = props_.find(key)->second;
  return mapbox::util::get<T>(val);
}

template <class T>
T& UnitDefinition::get(const std::string& key) {
  return mapbox::util::get<T>(props_[key]);
}

template <class T>
void UnitDefinition::set(const std::string& key, const T& value) {
  props_[key] = value;
}

}  // namespace internal

}  // namespace veles
#endif  // MAIN_FILE_LOADER_H_
