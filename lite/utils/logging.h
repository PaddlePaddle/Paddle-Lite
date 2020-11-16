// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * This file implements an lightweight alternative for glog, which is more
 * friendly for mobile.
 */
#pragma once

#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <assert.h>
#include <time.h>
#if !defined(_WIN32)
#include <sys/time.h>
#include <sys/types.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
extern struct timeval;
static int gettimeofday(struct timeval* tp, void* tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;

  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;

  return (0);
}
#endif

#include <cstdlib>
#include <cstring>
#include <string>
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"

#if defined(LITE_WITH_LOG) && defined(LITE_WITH_ANDROID)
#include <android/log.h>
// Android log macors
#define ANDROID_LOG_TAG "Paddle-Lite"
#define ANDROID_LOG_I(msg) \
  __android_log_print(ANDROID_LOG_INFO, ANDROID_LOG_TAG, "%s", msg)
#define ANDROID_LOG_W(msg) \
  __android_log_print(ANDROID_LOG_WARN, ANDROID_LOG_TAG, "%s", msg)
#define ANDROID_LOG_F(msg) \
  __android_log_print(ANDROID_LOG_FATAL, ANDROID_LOG_TAG, "%s", msg)
#endif

// NOLINTFILE()

// LOG()
#ifndef LITE_WITH_LOG
#define LOG(status) LOG_##status
#define LOG_INFO paddle::lite::Voidify()
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LOG_INFO
#define LOG_FATAL paddle::lite::VoidifyFatal()
#else
#define LOG(status) LOG_##status.stream()
#define LOG_INFO paddle::lite::LogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define LOG_ERROR LOG_INFO
#define LOG_WARNING \
  paddle::lite::LogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define LOG_FATAL \
  paddle::lite::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__)
#endif

#ifndef LITE_WITH_LOG
#define VLOG(level) paddle::lite::Voidify()
#else
// VLOG()
#define VLOG(level) \
  paddle::lite::VLogMessage(__FILE__, __FUNCTION__, __LINE__, level).stream()
#endif

// CHECK()
// clang-format off
#ifndef LITE_WITH_LOG
#define CHECK(x) if (!(x)) paddle::lite::VoidifyFatal()
#define _CHECK_BINARY(x, cmp, y) CHECK(x cmp y)
#else
#define CHECK(x) if (!(x)) paddle::lite::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__).stream() << "Check failed: " #x << ": " // NOLINT(*)
#define _CHECK_BINARY(x, cmp, y) CHECK((x cmp y)) << (x) << "!" #cmp << (y) << " " // NOLINT(*)
#endif

// clang-format on
#define CHECK_EQ(x, y) _CHECK_BINARY(x, ==, y)
#define CHECK_NE(x, y) _CHECK_BINARY(x, !=, y)
#define CHECK_LT(x, y) _CHECK_BINARY(x, <, y)
#define CHECK_LE(x, y) _CHECK_BINARY(x, <=, y)
#define CHECK_GT(x, y) _CHECK_BINARY(x, >, y)
#define CHECK_GE(x, y) _CHECK_BINARY(x, >=, y)

namespace paddle {
namespace lite {

#ifdef LITE_WITH_EXCEPTION
struct PaddleLiteException : public std::exception {
  const std::string exception_prefix = "Paddle-Lite C++ Exception: \n";
  std::string message;
  explicit PaddleLiteException(const char* detail) {
    message = exception_prefix + std::string(detail);
  }
  const char* what() const noexcept { return message.c_str(); }
};
#endif

#ifdef LITE_WITH_LOG
void gen_log(STL::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen = 40);

// LogMessage
class LogMessage {
 public:
  LogMessage(const char* file,
             const char* func,
             int lineno,
             const char* level = "I") {
    level_ = level;
    paddle::lite::gen_log(log_stream_, file, func, lineno, level);
  }

  ~LogMessage() {
    log_stream_ << '\n';
#ifdef LITE_WITH_ANDROID
    if (level_ == "I") {
      ANDROID_LOG_I(log_stream_.str().c_str());
    } else if (level_ == "W") {
      ANDROID_LOG_W(log_stream_.str().c_str());
    } else if (level_ == "F") {
      ANDROID_LOG_F(log_stream_.str().c_str());
    } else {
      fprintf(stderr, "Unsupported log level: %s\n", level_.c_str());
      assert(false);
    }
#endif
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  STL::ostream& stream() { return log_stream_; }

 protected:
  STL::stringstream log_stream_;
  std::string level_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

// LogMessageFatal
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file,
                  const char* func,
                  int lineno,
                  const char* level = "F")
      : LogMessage(file, func, lineno, level) {}

  ~LogMessageFatal()
#ifdef LITE_WITH_EXCEPTION
      noexcept(false)
#endif
  {
    log_stream_ << '\n';
#ifdef LITE_WITH_ANDROID
    ANDROID_LOG_F(log_stream_.str().c_str());
#endif
    fprintf(stderr, "%s", log_stream_.str().c_str());

#ifdef LITE_WITH_EXCEPTION
    throw PaddleLiteException(log_stream_.str().c_str());
#else
#ifndef LITE_ON_TINY_PUBLISH
    abort();
#else
    // If we decide whether the process exits according to the NDEBUG macro
    // definition, assert() can be used here.
    abort();
#endif
#endif
  }
};

// VLOG
class VLogMessage {
 public:
  VLogMessage(const char* file,
              const char* func,
              int lineno,
              const int32_t level_int = 0) {
    const char* GLOG_v = std::getenv("GLOG_v");
    GLOG_v_int = (GLOG_v && atoi(GLOG_v) > 0) ? atoi(GLOG_v) : 0;
    this->level_int = level_int;
    if (GLOG_v_int < level_int) {
      return;
    }
    const char* level = paddle::lite::to_string(level_int).c_str();
    paddle::lite::gen_log(log_stream_, file, func, lineno, level);
  }

  ~VLogMessage() {
    if (GLOG_v_int < this->level_int) {
      return;
    }
    log_stream_ << '\n';
#ifdef LITE_WITH_ANDROID
    ANDROID_LOG_I(log_stream_.str().c_str());
#endif
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  STL::ostream& stream() { return log_stream_; }

 protected:
  STL::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  VLogMessage(const VLogMessage&) = delete;
  void operator=(const VLogMessage&) = delete;
};
#else
class Voidify {
 public:
  Voidify() {}
  ~Voidify() {}

  template <typename T>
  Voidify& operator<<(const T& obj) {
    return *this;
  }
};

class VoidifyFatal : public Voidify {
 public:
#ifdef LITE_WITH_EXCEPTION
  ~VoidifyFatal() noexcept(false) { throw std::exception(); }
#else
  ~VoidifyFatal() {
    // If we decide whether the process exits according to the NDEBUG macro
    // definition, assert() can be used here.
    abort();
  }
#endif
};

#endif

}  // namespace lite
}  // namespace paddle
#endif
