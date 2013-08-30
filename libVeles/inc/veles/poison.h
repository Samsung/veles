/*! @file poison.h
 *  @brief Poison some functions which are not allowed in VELES code.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_VELES_POISON_H_
#define INC_VELES_POISON_H_

#if __GNUC__ >= 4
// Use the logging facilities instead
#pragma GCC poison printf
#endif

#endif  // INC_VELES_POISON_H_
