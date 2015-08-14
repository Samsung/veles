/*! @file logger.h
 *  @brief New file description.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
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

#ifndef INC_VELES_LOGGER_H_
#define INC_VELES_LOGGER_H_

#include <string>
#include <typeinfo>

#ifdef EINA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef EINA_STRIPPED
#include <Eina.h>
#else
#include <veles/eina_log.h>
#endif
#pragma GCC diagnostic pop
#endif

#ifdef EINA
#ifndef NO_POISON
#include <veles/poison.h>
#endif

#define EINA_COLOR_LIGHTVIOLET  "\033[35;1m"
#define EINA_COLOR_VIOLET       "\033[35m"

#define DBG(...) EINA_LOG_DOM_DBG(this->log_domain(), __VA_ARGS__)
#define INF(...) EINA_LOG_DOM_INFO(this->log_domain(), __VA_ARGS__)
#define WRN(...) EINA_LOG_DOM_WARN(this->log_domain(), __VA_ARGS__)
#define ERR(...) EINA_LOG_DOM_ERR(this->log_domain(), __VA_ARGS__)
#define CRT(...) EINA_LOG_DOM_CRIT(this->log_domain(), __VA_ARGS__)

#define DBGI(x, ...) EINA_LOG_DOM_DBG((x)->log_domain(), __VA_ARGS__)
#define INFI(x, ...) EINA_LOG_DOM_INFO((x)->log_domain(), __VA_ARGS__)
#define WRNI(x, ...) EINA_LOG_DOM_WARN((x)->log_domain(), __VA_ARGS__)
#define ERRI(x, ...) EINA_LOG_DOM_ERR((x)->log_domain(), __VA_ARGS__)
#define CRTI(x, ...) EINA_LOG_DOM_CRIT((x)->log_domain(), __VA_ARGS__)

#define DBGS(x, ...) EINA_LOG_DOM_DBG(x::log_domain(), __VA_ARGS__)
#define INFS(x, ...) EINA_LOG_DOM_INFO(x::log_domain(), __VA_ARGS__)
#define WRNS(x, ...) EINA_LOG_DOM_WARN(x::log_domain(), __VA_ARGS__)
#define ERRS(x, ...) EINA_LOG_DOM_ERR(x::log_domain(), __VA_ARGS__)
#define CRTS(x, ...) EINA_LOG_DOM_CRIT(x::log_domain(), __VA_ARGS__)

#else

#include <stdio.h>

#define FALLBACK_LOG(...) { fprintf(stderr, __VA_ARGS__); \
                            fprintf(stderr, "\n"); }

#define DBG(...) FALLBACK_LOG(__VA_ARGS__)
#define INF(...) FALLBACK_LOG(__VA_ARGS__)
#define WRN(...) FALLBACK_LOG(__VA_ARGS__)
#define ERR(...) FALLBACK_LOG(__VA_ARGS__)
#define CRT(...) FALLBACK_LOG(__VA_ARGS__)

#define DBGI(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define INFI(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define WRNI(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define ERRI(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define CRTI(x, ...) FALLBACK_LOG(__VA_ARGS__)

#define DBGS(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define INFS(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define WRNS(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define ERRS(x, ...) FALLBACK_LOG(__VA_ARGS__)
#define CRTS(x, ...) FALLBACK_LOG(__VA_ARGS__)

#endif

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {

/// @brief Enables logging with different levels and domains for inherited
/// classes.
/// @details How to use: simply replace printf()-s with DBG, INF, WRN, ERR or
/// CRT.
/// Rules of usage:
/// 1) Never place '\n' or '.' at the end of the message.
/// 2) Conform to the meaning of each logging level (DBG, INF, WRN, ERR and
///    CRT).
/// 3) Protected inheritance should be preferred; if any other class needs to
///    print logs on behalf on this one, use public.
class Logger {
 public:
  enum Color {
    COLOR_LIGHTRED = 0,
    COLOR_RED,
    COLOR_LIGHTBLUE,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_WHITE,
    COLOR_LIGHTCYAN,
    COLOR_CYAN,
    COLOR_RESET,
    COLOR_HIGH,
    COLOR_LIGHTVIOLET,
    COLOR_VIOLET,
  };


  // The following defines must not be converted to static const members of Logger
  // due to the undefined order in which static constructors are invoked.
  #ifdef EINA
  #define kDefaultLoggerColor EINA_COLOR_WHITE
  #else
  #define kDefaultLoggerColor "\033[37;1m"
  #endif

  Logger(const std::string &domain = "VELES",
         const std::string &color = kDefaultLoggerColor
#ifdef EINA
         , bool suppressLoggingInitialized = true
#endif
         ) noexcept;

  Logger(const Logger& other) noexcept;

  Logger& operator=(const Logger& other) noexcept;

  Logger& operator=(Logger&& other) noexcept;

  virtual ~Logger();

  int log_domain() const noexcept;

  std::string domain_str() const noexcept;

  void set_domain_str(const std::string &value) noexcept;

  std::string color() const noexcept;

  void set_color(const std::string &value) noexcept;

  static std::string GetColorByIndex(unsigned index) noexcept;

 protected:
  static std::string Demangle(const char *symbol) noexcept;

 private:
  int log_domain_;
  std::string domain_str_;
  std::string color_;
#ifdef EINA
  bool suppress_logging_initialized_;

  void InitializeEina() noexcept;
  void DisposeEina() noexcept;
#endif
};

template <class T, unsigned C = 0>
class DefaultLogger : public Logger {
 public:
  DefaultLogger() : Logger(StripVelesNamespace(Demangle(typeid(T).name())),
                           GetColorByIndex(C)) {
  }

 private:
  static std::string StripVelesNamespace(std::string&& symbol) noexcept {
    std::string res(std::forward<std::string>(symbol));
    size_t pos;
    while ((pos = res.find("veles::")) != std::string::npos) {
      res = res.erase(pos, 7);
    }
    return res;
  }
};

}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_VELES_LOGGER_H_
