/*! @file dummy_main.c
 *  @brief New file description.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "eina_log.h"

int main() {
  eina_log_init();
  EINA_LOG_DBG("Hello, World!");
  eina_log_shutdown();
  return 0;
}
