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
#include "inc/veles/workflow.h"

namespace Veles {

size_t Workflow::MaxUnitSize() const noexcept {
  auto max_func = [](size_t curr, std::shared_ptr<Unit> unit) {
    return std::max(curr, unit->OutputCount());
  };
  return std::max(
      std::accumulate(units_.begin(), units_.end(),
                      static_cast<size_t>(0), max_func),
      !units_.empty() ? units_.front()->OutputCount() : 0);
}

void* Workflow::MallocF(size_t length) {
  void *ptr;
  return posix_memalign(&ptr, 64, length * sizeof(float)) == 0 ? ptr : nullptr;
}

}  // namespace Veles

