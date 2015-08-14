/*! @file workflow.cc
 *  @brief VELES workflow
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "inc/veles/workflow.h"

namespace veles {

std::shared_ptr<void> Workflow::GetProperty(const std::string& name) {
  auto it = props_.find(name);
  return it != props_.end() ? it->second : nullptr;
}

void Workflow::SetProperties(const PropertiesTable& table) {
  props_ = table;
}

std::shared_ptr<Unit> Workflow::Get(size_t index) const {
  if (index >= Size()) {
    throw std::out_of_range("index");
  }
  return units_[index];
}

size_t Workflow::MaxUnitSize() const noexcept {
  auto max_func = [](size_t curr, std::shared_ptr<Unit> unit) {
    return std::max(curr, unit->OutputCount());
  };
  return std::accumulate(units_.begin(), units_.end(),
                         !units_.empty() ? units_.front()->InputCount() : 0,
                         max_func);
}

float* Workflow::mallocf(size_t length) {
#ifndef __ANDROID__
  void *ptr;
  return posix_memalign(&ptr, 64, length * sizeof(float)) == 0 ?
      static_cast<float*>(ptr) : nullptr;
#else
  return reinterpret_cast<float*>(memalign(64, length * sizeof(float)));
#endif
}

}  // namespace veles

