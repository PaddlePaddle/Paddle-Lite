/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#ifdef PADDLE_MOBILE_DEBUG
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#endif
#ifdef ANDROID
#include <android/log.h>
#endif

namespace paddle_mobile {

#ifdef PADDLE_MOBILE_DEBUG

#ifdef ANDROID

static const char *ANDROID_LOG_TAG =
    "paddle_mobile LOG built on " __DATE__ " " __TIME__;
#ifdef PADDLE_ENABLE_COLORABLE_LOG
#define PADDLE_RED "\033[1;31;40m"
#define PADDLE_GREEN "\033[1;32;40m"
#define PADDLE_YELLOW "\033[1;33;40m"
#define PADDLE_LIGHT_RED "\033[1;35;40m"
#define PADDLE_BLUE "\033[1;34;40m"
#define PADDLE_WHITE "\033[1;37;40m"
#define PADDLE_CONON "\033[0m"
#else
#define PADDLE_RED ""
#define PADDLE_GREEN ""
#define PADDLE_YELLOW ""
#define PADDLE_LIGHT_RED ""
#define PADDLE_BLUE ""
#define PADDLE_WHITE ""
#define PADDLE_CONON ""
#endif
#define ANDROIDLOGI(...)                                               \
  __android_log_print(ANDROID_LOG_INFO, ANDROID_LOG_TAG, __VA_ARGS__); \
  fprintf(stderr, PADDLE_YELLOW "%s\n" PADDLE_CONON, __VA_ARGS__);     \
  fflush(stderr)
#define ANDROIDLOGW(...)                                               \
  __android_log_print(ANDROID_LOG_WARN, ANDROID_LOG_TAG, __VA_ARGS__); \
  fprintf(stderr, PADDLE_LIGHT_RED "%s\n" PADDLE_CONON, __VA_ARGS__);  \
  fflush(stderr)
#define ANDROIDLOGD(...)                                                \
  __android_log_print(ANDROID_LOG_DEBUG, ANDROID_LOG_TAG, __VA_ARGS__); \
  fprintf(stderr, PADDLE_WHITE "%s\n" PADDLE_CONON, __VA_ARGS__);       \
  fflush(stderr)
#define ANDROIDLOGE(...)                                                \
  __android_log_print(ANDROID_LOG_ERROR, ANDROID_LOG_TAG, __VA_ARGS__); \
  fprintf(stderr, PADDLE_RED "%s\n" PADDLE_CONON, __VA_ARGS__);         \
  fflush(stderr)
#define ANDROIDLOGV(...)                                                  \
  __android_log_print(ANDROID_LOG_VERBOSE, ANDROID_LOG_TAG, __VA_ARGS__); \
  fprintf(stderr, PADDLE_GREEN "%s\n" PADDLE_CONON, __VA_ARGS__);         \
  fflush(stderr)
#else
#define ANDROIDLOGI(...)
#define ANDROIDLOGW(...)
#define ANDROIDLOGD(...)
#define ANDROIDLOGE(...)
#define ANDROIDLOGV(...)

#endif

enum LogLevel {
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_VERBOSE,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4,
  kNO_LOG,
};

// log level
static LogLevel log_level = kLOG_DEBUG4;

static std::vector<std::string> logs{"ERROR  ", "WARNING", "INFO   ", "VERBOSE",
                                     "DEBUG  ", "DEBUG1 ", "DEBUG2 ", "DEBUG3 ",
                                     "DEBUG4 ", "NO     "};
struct ToLog;
struct Print;

struct Print {
  friend struct ToLog;

  template <typename T>
  Print &operator<<(T const &value) {
    buffer_ << value;
    return *this;
  }

 private:
  void print(LogLevel level) {
    // buffer_ << std::endl;
    if (level == kLOG_ERROR) {
#ifdef ANDROID
      ANDROIDLOGE(buffer_.str().c_str());
#else
      std::cerr << buffer_.str() << std::endl;
#endif
    } else if (level == kLOG_INFO) {
#ifdef ANDROID
      ANDROIDLOGI(buffer_.str().c_str());
#else
      std::cerr << buffer_.str() << std::endl;
#endif
    } else if (level == kLOG_VERBOSE) {
#ifdef ANDROID
      ANDROIDLOGV(buffer_.str().c_str());
#else
      std::cerr << buffer_.str() << std::endl;
#endif
    } else if (level == kLOG_WARNING) {
#ifdef ANDROID
      ANDROIDLOGW(buffer_.str().c_str());
#else
      std::cerr << buffer_.str() << std::endl;
#endif
    } else {
#ifdef ANDROID
      ANDROIDLOGD(buffer_.str().c_str());
#else
      std::cout << buffer_.str() << std::endl;
#endif
    }
  }
  std::ostringstream buffer_;
};

struct ToLog {
  explicit ToLog(LogLevel level = kLOG_DEBUG, const std::string &info = "")
      : level_(level) {
    unsigned blanks =
        (unsigned)(level > kLOG_DEBUG ? (level - kLOG_DEBUG) * 4 : 1);
    printer_ << logs[level] << " " << info << ":" << std::string(blanks, ' ');
  }

  template <typename T>
  ToLog &operator<<(T const &value) {
    printer_ << value;
    return *this;
  }

  ~ToLog() { printer_.print(level_); }

 private:
  LogLevel level_;
  Print printer_;
};

#define LOG(level)                                                           \
  if (level > paddle_mobile::log_level) {                                    \
    /* NOLINTNEXTLINE */                                                     \
  } else                                                                     \
    paddle_mobile::ToLog(                                                    \
        level, static_cast<const std::stringstream &>(                       \
                   std::stringstream()                                       \
                   << "[file: "                                              \
                   << (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) \
                                              : __FILE__)                    \
                   << "] [line: " << __LINE__ << "] ")                       \
                   .str())

#define DLOG                                                          \
  if (paddle_mobile::kLOG_DEBUG > paddle_mobile::log_level) {         \
    /* NOLINTNEXTLINE */                                              \
  } else                                                              \
    paddle_mobile::ToLog(                                             \
        paddle_mobile::kLOG_DEBUG,                                    \
        static_cast<const std::stringstream &>(                       \
            std::stringstream()                                       \
            << "[file: "                                              \
            << (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) \
                                       : __FILE__)                    \
            << "] [line: " << __LINE__ << "] ")                       \
            .str())

#define LOGF(level, format, ...)          \
  if (level > paddle_mobile::log_level) { \
    /* NOLINTNEXTLINE */                  \
  } else                                  \
    printf(format, ##__VA_ARGS__)

#define DLOGF(format, ...)                                    \
  if (paddle_mobile::kLOG_DEBUG > paddle_mobile::log_level) { \
    /* NOLINTNEXTLINE */                                      \
  } else                                                      \
    printf(format, ##__VA_ARGS__)

#else

#define ANDROIDLOGI(...)
#define ANDROIDLOGW(...)
#define ANDROIDLOGD(...)
#define ANDROIDLOGE(...)
#define ANDROIDLOGV(...)

enum LogLevel {
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_VERBOSE,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4,
  kNO_LOG
};

struct ToLog;
struct Print {
  friend struct ToLog;
  template <typename T>
  Print &operator<<(T const &value) {
    return *this;
  }
};

struct ToLog {
  explicit ToLog(LogLevel level) {}

  template <typename T>
  ToLog &operator<<(T const &value) {
    return *this;
  }
};

#define LOG(level)       \
  if (true) {            \
    /* NOLINTNEXTLINE */ \
  } else                 \
    paddle_mobile::ToLog(level)

#define DLOG             \
  if (true) {            \
    /* NOLINTNEXTLINE */ \
  } else                 \
    paddle_mobile::ToLog(paddle_mobile::kLOG_DEBUG)

#define LOGF(level, format, ...)

#define DLOGF(format, ...)

#endif

template <typename T>
Print &operator<<(Print &printer, const std::vector<T> &v) {
  printer << "[ ";

  for (int i = 0; i < v.size(); ++i) {
    const auto &value = v[i];
    printer << value << " ";
    if (i % 10 == 9) {
      printer << "\n";
    }
  }
  printer << " ]";
  return printer;
}

}  // namespace paddle_mobile
