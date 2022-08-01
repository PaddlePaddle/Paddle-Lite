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

#include "utility/logging.h"
#include <iomanip>
#include "utility/micros.h"

namespace nnadapter {
namespace logging {

void gen_log(std::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen = 40) {
  const int len = strlen(file);

  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
#if defined(_WIN32)
  localtime_s(&tm_time, &timestamp);
#else
  localtime_r(&timestamp, &tm_time);
#endif
  struct timeval tv;
  gettimeofday(&tv, NULL);

  // print date / time
  log_stream_ << '[' << level << ' ' << std::setw(2) << 1 + tm_time.tm_mon
              << '/' << std::setw(2) << tm_time.tm_mday << ' ' << std::setw(2)
              << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
              << std::setw(2) << tm_time.tm_sec << '.' << std::setw(3)
              << tv.tv_usec / 1000 << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func
                << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}

NNADAPTER_EXPORT LogMessage::LogMessage(const char* file,
                                        const char* func,
                                        int lineno,
                                        const char* level)
    : level_(level) {
  gen_log(log_stream_, file, func, lineno, level);
}

NNADAPTER_EXPORT LogMessage::~LogMessage() {
  log_stream_ << '\n';
#if defined(ANDROID) || defined(__ANDROID__)
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

NNADAPTER_EXPORT LogMessageFatal::~LogMessageFatal() noexcept(false) {
  log_stream_ << '\n';
#if defined(ANDROID) || defined(__ANDROID__)
  ANDROID_LOG_F(log_stream_.str().c_str());
#endif
  fprintf(stderr, "%s", log_stream_.str().c_str());
  throw Exception(log_stream_.str().c_str());
  abort();
}

NNADAPTER_EXPORT VLogMessage::VLogMessage(const char* file,
                                          const char* func,
                                          int lineno,
                                          const int32_t level_int) {
  const char* GLOG_v = std::getenv("GLOG_v");
  GLOG_v_int = (GLOG_v && atoi(GLOG_v) > 0) ? atoi(GLOG_v) : 0;
  this->level_int = level_int;
  if (GLOG_v_int < level_int) {
    return;
  }
  const char* level = std::to_string(level_int).c_str();
  gen_log(log_stream_, file, func, lineno, level);
}

NNADAPTER_EXPORT VLogMessage::~VLogMessage() {
  if (GLOG_v_int < this->level_int) {
    return;
  }
  log_stream_ << '\n';
#if defined(ANDROID) || defined(__ANDROID__)
  ANDROID_LOG_I(log_stream_.str().c_str());
#endif
  fprintf(stderr, "%s", log_stream_.str().c_str());
}

}  // namespace logging
}  // namespace nnadapter
