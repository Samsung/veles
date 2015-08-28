/*! @file workflow_loader.cc
 *  @brief Implementation of WorkflowLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>, Bulychev Egor <e.bulychev@samsung.com>
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

#include "veles/workflow_loader.h"
#include "inc/veles/unit_factory.h"
#include "src/workflow_archive.h"

namespace veles {

WorkflowLoader::WorkflowLoader() {
}

Workflow WorkflowLoader::Load(const std::string& file_name,
                              const std::shared_ptr<Engine>& engine) {
  auto war = internal::WorkflowArchive::Load(file_name);
  auto& wdef = war->workflow_definition();
  auto head = CreateUnit(war, wdef.start(), engine, nullptr);
  return Workflow(wdef.name(), wdef.checksum(), head, engine);
}

namespace {

class PropertyVisitor {
 public:
  PropertyVisitor(const std::shared_ptr<internal::WorkflowArchive>& archive,
          veles::Property* property) : archive_(archive), property_(property) {
  }

  template <class T>
  void operator()(const T& value) {
    *property_ = value;
  }

  void operator()(const internal::NumpyArrayReference& ref) {
    *property_ = PackagedNumpyArray(ref, archive_);
  }

 private:
  const std::shared_ptr<internal::WorkflowArchive>& archive_;
  Property* property_;
};

}

std::shared_ptr<Unit> WorkflowLoader::CreateUnit(
    const std::shared_ptr<internal::WorkflowArchive> war,
    const std::shared_ptr<internal::UnitDefinition>& udef,
    const std::shared_ptr<Engine>& engine,
    std::shared_ptr<Unit> parent) const {
  auto ctor = UnitFactory::Instance()[udef->uuid_str()];
  if (ctor == nullptr) {
    throw UnitNotFoundException(udef->uuid_str(), udef->name());
  }
  auto unit = ctor(engine);

  for (auto& prop : udef->properties()) {
    Property value;
    PropertyVisitor visitor(war, &value);
    internal::Property::visit(prop.second, visitor);
    unit->SetParameter(prop.first, value);
  }
  if (parent) {
    unit->LinkFrom(parent);
  }
  for (auto& childdef : udef->links()) {
    CreateUnit(war, childdef, engine, unit);
  }
  return unit;
}

}  // namespace veles
