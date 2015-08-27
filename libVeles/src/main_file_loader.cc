/*! @file main_file_loader.cc
 *  @brief Implementation of MainFileLoader class.
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

#include "src/main_file_loader.h"
#include <algorithm>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <rapidjson/document.h>
#pragma GCC diagnostic pop

namespace veles {

namespace internal {

UnitDefinition::UnitDefinition(
    const std::string& name, const std::string& uuid)
    : name_(name) {
  std::string hex(uuid);
  hex.erase(std::remove(hex.begin(), hex.end(), '-'), hex.end());
  for (int i = 0; i < 32; i+=2) {
    uuid_[i >> 1] = std::stoi(hex.substr(i, 2), nullptr, 16);
  }
}

std::string UnitDefinition::uuid_str() const noexcept {
  std::string res(36, 0);
  static constexpr char hexmap[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                    '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  int offset = 0;
  for (int i = 0; i < static_cast<int>(sizeof(uuid_)); i++) {
    res[i * 2 + offset] = hexmap[uuid_[i] >> 4];
    res[i * 2 + offset + 1] = hexmap[uuid_[i] & 0xF];
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      res[i * 2 + ++offset + 1] = '-';
    }
  }
  return res;
}

std::vector<std::string> UnitDefinition::PropertyNames() const noexcept {
  std::vector<std::string> keys;
  keys.reserve(props_.size());
  for (auto& it : props_) {
    keys.push_back(it.first);
  }
  return keys;
}

void UnitDefinition::Link(std::shared_ptr<UnitDefinition> def) {
  links_.push_back(def);
}

WorkflowDefinition::WorkflowDefinition(
    const std::string& checksum, const std::string& name,
    std::shared_ptr<UnitDefinition> start)
    : checksum_(checksum), name_(name), start_(start) {
}

namespace {

template <class T>
class InputStreamWrapper {
 public:
  typedef T Ch;
  InputStreamWrapper(std::basic_istream<T>* in) : in_(*in) {}

  InputStreamWrapper(const InputStreamWrapper&) = delete;
  InputStreamWrapper& operator=(const InputStreamWrapper&) = delete;

  T Peek() const {
    int c = in_.peek();
    return c == std::char_traits<T>::eof()? 0 : static_cast<T>(c);
  }

  T Take() {
    int c = in_.get();
    return c == std::char_traits<T>::eof()? 0 : static_cast<T>(c);
  }

  size_t Tell() const { return static_cast<size_t>(in_.tellg()); }
  T* PutBegin() { assert(false); return 0; }
  void Put(T) { assert(false); }
  void Flush() { assert(false); }
  size_t PutEnd(T*) { assert(false); return 0; }

 private:
  std::istream& in_;
};

}

WorkflowDefinition MainFileLoader::Load(std::istream* src) {
  rapidjson::Document doc;
  InputStreamWrapper<char> srcwrapper(src);
  doc.ParseStream(srcwrapper);
  auto& units = doc["units"];
  std::vector<std::shared_ptr<UnitDefinition>> udefs(units.Size());
  for (int i = static_cast<int>(units.Size()) - 1; i >= 0; i--) {
    auto& unit = units[i];
    auto uuid = unit["class"]["uuid"].GetString();
    auto udef = std::make_shared<UnitDefinition>(
        unit["class"]["name"].GetString(), uuid);
    DBG("Unit %s: %s", uuid, udef->name().c_str());
    udefs[i] = udef;
    auto& props = unit["data"];
    for (auto pit = props.MemberBegin(); pit != props.MemberEnd(); ++pit) {
      switch (pit->value.GetType()) {
        case rapidjson::kFalseType:
        case rapidjson::kTrueType:
          udef->set(pit->name.GetString(), pit->value.GetBool());
          break;
        case rapidjson::kStringType: {
          std::string value = pit->value.GetString();
          if (value.size() > 1 && value[0] == '@') {
            value = value.substr(1);
            if (value[1] != '@') {
              udef->set(pit->name.GetString(),
                        NumpyArrayReference(value + ".npy"));
              break;
            }
          }
          udef->set(pit->name.GetString(), value);
          break;
        }
        case rapidjson::kNumberType:
          if (pit->value.IsDouble()) {
            udef->set(pit->name.GetString(),
                      static_cast<float>(pit->value.GetDouble()));
          } else if (pit->value.IsInt()) {
            udef->set(pit->name.GetString(), pit->value.GetInt());
          } else {
            WRN("Unsupported property type: int64");
          }
          break;
        default:
          WRN("Unsupported property type: %d", pit->value.GetType());
          break;
      }
    }
    auto& links = unit["links"];
    for (rapidjson::SizeType l = 0; l < links.Size(); l++) {
      udef->Link(udefs[links[l].GetInt()]);
    }
  }
  auto checksum = doc["checksum"].GetString();
  auto name = doc["workflow"].GetString();
  DBG("Workflow %s: %s", checksum, name);
  return { checksum, name, udefs[0] };
}

}  // namespace internal

}  // namespace veles
