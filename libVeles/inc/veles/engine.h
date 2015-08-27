/*! @file engine.h
 *  @brief New file description.
 *  @author markhor
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_VELES_ENGINE_H_
#define INC_VELES_ENGINE_H_

#include <functional>

namespace veles {

class Engine {
 public:
  typedef std::function<void(void)> Callable;

  virtual ~Engine() = default;

  virtual void Schedule(const Callable& callable) = 0;

  virtual void Finish() = 0;
};

}  // namespace veles
#endif  // INC_VELES_ENGINE_H_
