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
 * This file implements an lightweight alternative for glog, which is from
 * PaddleLite
 */

#pragma once

#include <assert.h>
#include <time.h>
#if !defined(_WIN32)
#include <sys/time.h>
#include <sys/types.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#undef min
#undef max
extern struct timeval;
static int gettimeofday(struct timeval* tp, void* tzp) {
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  tp->tv_sec = now.QuadPart / freq.QuadPart;
  tp->tv_usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
  return (0);
}
#endif

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
// Android log macors
#define ANDROID_LOG_TAG "FAKE_DDK"
#define ANDROID_LOG_I(msg) \
  __android_log_print(ANDROID_LOG_INFO, ANDROID_LOG_TAG, "%s", msg)
#define ANDROID_LOG_W(msg) \
  __android_log_print(ANDROID_LOG_WARN, ANDROID_LOG_TAG, "%s", msg)
#define ANDROID_LOG_F(msg) \
  __android_log_print(ANDROID_LOG_FATAL, ANDROID_LOG_TAG, "%s", msg)
#endif

// FAKE_DDK_LOG()
#define FAKE_DDK_LOG(status) FAKE_DDK_LOG_##status.stream()
#define FAKE_DDK_LOG_INFO \
  fake_ddk::logging::LogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define FAKE_DDK_LOG_ERROR FAKE_DDK_LOG_INFO
#define FAKE_DDK_LOG_WARNING \
  fake_ddk::logging::LogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define FAKE_DDK_LOG_FATAL \
  fake_ddk::logging::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__)

// FAKE_DDK_VLOG()
#define FAKE_DDK_VLOG(level)                                              \
  fake_ddk::logging::VLogMessage(__FILE__, __FUNCTION__, __LINE__, level) \
      .stream()

// FAKE_DDK_CHECK()
// clang-format off
#define FAKE_DDK_CHECK(x) if (!(x)) fake_ddk::logging::LogMessageFatal(__FILE__, __FUNCTION__, __LINE__).stream() << "Check failed: " #x << ": " // NOLINT(*)
#define _FAKE_DDK_CHECK_BINARY(x, cmp, y) FAKE_DDK_CHECK((x cmp y)) << (x) << "!" #cmp << (y) << " " // NOLINT(*)

// clang-format on
#define FAKE_DDK_CHECK_EQ(x, y) _FAKE_DDK_CHECK_BINARY(x, ==, y)
#define FAKE_DDK_CHECK_NE(x, y) _FAKE_DDK_CHECK_BINARY(x, !=, y)
#define FAKE_DDK_CHECK_LT(x, y) _FAKE_DDK_CHECK_BINARY(x, <, y)
#define FAKE_DDK_CHECK_LE(x, y) _FAKE_DDK_CHECK_BINARY(x, <=, y)
#define FAKE_DDK_CHECK_GT(x, y) _FAKE_DDK_CHECK_BINARY(x, >, y)
#define FAKE_DDK_CHECK_GE(x, y) _FAKE_DDK_CHECK_BINARY(x, >=, y)

namespace fake_ddk {
namespace logging {

struct Exception : public std::exception {
  const std::string exception_prefix = "FakeDDK C++ Exception: \n";
  std::string message;
  explicit Exception(const char* detail) {
    message = exception_prefix + std::string(detail);
  }
  const char* what() const noexcept { return message.c_str(); }
};

class LogMessage {
 public:
  LogMessage(const char* file,
             const char* func,
             int lineno,
             const char* level = "I");
  ~LogMessage();

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  std::string level_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file,
                  const char* func,
                  int lineno,
                  const char* level = "F")
      : LogMessage(file, func, lineno, level) {}
  ~LogMessageFatal() noexcept(false);
};

class VLogMessage {
 public:
  VLogMessage(const char* file,
              const char* func,
              int lineno,
              const int32_t level_int = 0);
  ~VLogMessage();

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  VLogMessage(const VLogMessage&) = delete;
  void operator=(const VLogMessage&) = delete;
};

}  // namespace logging
}  // namespace fake_ddk
