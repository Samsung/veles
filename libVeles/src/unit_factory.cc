/*! @file unit_factory.cc
 *  @brief Implements veles::UnitFactory class.
 *  @author markhor
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013 Samsung R&D Institute Russia
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

#include "inc/veles/unit_factory.h"
#include <cstdio>
#include <string>

namespace veles {

UnitFactory::UnitFactory()
    : Logger("UnitFactory", GetColorByIndex(Logger::COLOR_LIGHTBLUE)) {
}

const UnitFactory& UnitFactory::Instance() {
  return Origin();
}

UnitFactory& UnitFactory::Origin() {
  static UnitFactory instance;
  return instance;
}

UnitFactory::UnitConstructor UnitFactory::operator[](
    const std::string& uuid) const {
  auto f = map_.find(uuid);
  if (f != map_.end()) {
    return f->second;
  }
  return nullptr;
}

void UnitFactory::PrintRegisteredUnits() const {
  for (auto& tit : map_) {
    INF("%s", tit.first.c_str());
  }
}

}  // namespace veles
