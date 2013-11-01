/*! @file logger.cc
 *  @brief New file description.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "inc/veles/logger.h"
#include <assert.h>
#include <cxxabi.h>
#include <string.h>

namespace veles {

// The following defines must not be converted to static const members of Logger
// due to the undefined order in which static constructors are invoked.
#ifdef EINA
#define kDefaultLoggerColor EINA_COLOR_WHITE
#else
#define kDefaultLoggerColor "\033[37;1m"
#endif


#define kCommonDomain ""
#define kUnintializedLogDomain_ (-1)

Logger::Logger(const std::string &domain = "VELES",
               const std::string &color = kDefaultLoggerColor
#ifdef EINA
               , bool suppressLoggingInitialized
#endif
               ) noexcept
    : log_domain_(kUnintializedLogDomain_)
    , domain_str_(domain)
    , color_(color)
#ifdef EINA
    , suppress_logging_initialized_(suppressLoggingInitialized)
#endif
    {
#ifdef EINA
  InitializeEina();
#endif
}

Logger::Logger(const Logger& other) noexcept
    : log_domain_(kUnintializedLogDomain_)
    , domain_str_(other.domain_str_)
    , color_(other.color_)
#ifdef EINA
    , suppress_logging_initialized_(other.suppress_logging_initialized_)
#endif
    {
#ifdef EINA
  InitializeEina();
#endif
}

Logger& Logger::operator=(const Logger& other) noexcept {
  log_domain_ = kUnintializedLogDomain_;
  domain_str_ = (other.domain_str_);
  color_ = other.color_;
#ifdef EINA
  suppress_logging_initialized_ = other.suppress_logging_initialized_;
  InitializeEina();
#endif
  return *this;
}

Logger::~Logger() {
#ifdef EINA
  DisposeEina();
#endif
}

#ifdef EINA

void Logger::InitializeEina() noexcept {
  DisposeEina();
#ifndef EINA_STRIPPED
  eina_init();
#else
  eina_log_init();
#endif
  eina_log_threads_enable();
  int len = strlen(kCommonDomain) + strlen(domain_str_.c_str()) + 1;
  char *fullDomain = new char[len];
  snprintf(fullDomain, len, "%s%s", kCommonDomain, domain_str_.c_str());
  log_domain_ = eina_log_domain_register(fullDomain, color_.c_str());
  if (log_domain_ < 0) {
    int message_len = len + 128;
    char *message = new char[message_len];
    snprintf(message, message_len, "%s%s%s",
            "could not register ", fullDomain, " log domain.");
    EINA_LOG_DOM_ERR(EINA_LOG_DOMAIN_GLOBAL, "%s", message);
    log_domain_ = EINA_LOG_DOMAIN_GLOBAL;
  } else {
    if (!suppress_logging_initialized_) {
      DBG("Logging was initialized with domain %i.",
          log_domain_);
    }
  }
  delete[] fullDomain;
}

void Logger::DisposeEina() noexcept {
  if (log_domain_ != kUnintializedLogDomain_ &&
      log_domain_ != EINA_LOG_DOMAIN_GLOBAL) {
    if (!suppress_logging_initialized_) {
      DBG("Domain %i is not registered now", log_domain_);
    }
    eina_log_domain_unregister(log_domain_);
    log_domain_ = kUnintializedLogDomain_;
  }
}

#endif

int Logger::log_domain() const noexcept {
#ifdef EINA
  assert(log_domain_ != kUnintializedLogDomain_);
#endif
  return log_domain_;
}

std::string Logger::domain_str() const noexcept {
  return domain_str_;
}

void Logger::set_domain_str(const std::string &value) noexcept {
  domain_str_ = value;
#ifdef EINA
  InitializeEina();
#endif
}

std::string Logger::color() const noexcept {
  return color_;
}

void Logger::set_color(const std::string &value) noexcept {
  color_ = value;
#ifdef EINA
  InitializeEina();
#endif
}

std::string Logger::Demangle(const char *symbol) noexcept {
  char result[1024];
  size_t length = sizeof(result);
  int status = 1;
  abi::__cxa_demangle(symbol, result, &length, &status);
  if (status == 0) {
    return result;
  }
  return symbol;
}

std::string Logger::GetColorByIndex(unsigned
#ifdef EINA
                                    index
#endif
                                    ) noexcept {
#ifdef EINA
  switch (index) {
    case COLOR_LIGHTRED:
      return EINA_COLOR_LIGHTRED;
    case COLOR_RED:
      return EINA_COLOR_RED;
    case COLOR_LIGHTBLUE:
      return EINA_COLOR_LIGHTBLUE;
    case COLOR_BLUE:
      return EINA_COLOR_BLUE;
    case COLOR_GREEN:
      return EINA_COLOR_GREEN;
    case COLOR_YELLOW:
      return EINA_COLOR_YELLOW;
    case COLOR_ORANGE:
      return EINA_COLOR_ORANGE;
    case COLOR_WHITE:
      return EINA_COLOR_WHITE;
    case COLOR_LIGHTCYAN:
      return EINA_COLOR_LIGHTCYAN;
    case COLOR_CYAN:
      return EINA_COLOR_CYAN;
    case COLOR_RESET:
      return EINA_COLOR_RESET;
    case COLOR_HIGH:
      return EINA_COLOR_HIGH;
    case COLOR_LIGHTVIOLET:
      return EINA_COLOR_LIGHTVIOLET;
    case COLOR_VIOLET:
      return EINA_COLOR_VIOLET;
    default:
      return "";
  }
#else
  return "";
#endif
}

}  // namespace SoundFeatureExtraction
