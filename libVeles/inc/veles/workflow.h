/*! @file workflow.h
 *  @brief VELES Workflow
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_WORKFLOW_H_
#define INC_WORKFLOW_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <veles/unit.h>

namespace Veles {

/** @brief VELES workflow */
class Workflow {
 public:
  virtual ~Workflow() = default;
  /** @brief Appends a unit to the end of workflow
   *  @param unit VELES unit
   */
  void AddUnit(std::shared_ptr<Unit> unit) {
    units_.push_back(unit);
  }

  /** @brief Clears the Workflow
   */
  void Clear() {
    units_.clear();
  }

  /** @brief Executes the workflow
   *  @param begin Iterator to the first element of initial data
   *  @param end Iterator to the end of initial data
   *  @param out Output iterator for the result
   */
  template<class InputIterator, class OutputIterator>
  void Execute(InputIterator begin, InputIterator end,
                         OutputIterator out) const {
    size_t max_size = MaxUnitSize();
    std::unique_ptr<float[], void (*)(void*)> input(
        MallocF(max_size), std::free);
    std::unique_ptr<float[], void (*)(void*)> output(
        MallocF(max_size), std::free);
    std::copy(begin, end, input.get());

    float* curr_in = input.get();
    float* curr_out = output.get();

    if (!units_.empty()) {
      for (const auto& unit : units_) {
        unit->Execute(curr_in, curr_out);
        std::swap(curr_in, curr_out);
      }
      std::copy(curr_in, curr_in + units_.back()->OutputCount(), out);
    }
  }

 private:
  /** @brief Get maximum input and output size of containing units
   *  @return Maximum size
   */
  size_t MaxUnitSize() const noexcept;
  static void* MallocF(size_t length);

  std::vector<std::shared_ptr<Unit>> units_;
};

}  // namespace Veles

#endif  // INC_WORKFLOW_H_
