/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#ifdef PADDLE_MOBILE_DEBUG

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace paddle_mobile {

enum LogLevel {
  kNO_LOG,
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4
};

// log level
static LogLevel log_level = kLOG_DEBUG4;

static std::vector<std::string> logs{"NO",      "ERROR ",  "WARNING",
                                     "INFO   ", "DEBUG  ", "DEBUG1 ",
                                     "DEBUG2 ", "DEBUG3 ", "DEBUG4 "};

struct ToLog;

struct Print {
  friend struct ToLog;
  template <typename T> Print &operator<<(T const &value) {
    buffer_ << value;
    return *this;
  }

private:
  void print(LogLevel level) {
    buffer_ << std::endl;
    if (level == kLOG_ERROR) {
      std::cerr << buffer_.str();
    } else {
      std::cout << buffer_.str();
    }
  }
  std::ostringstream buffer_;
};

struct ToLog {
  ToLog(LogLevel level = kLOG_DEBUG, const std::string &info = "")
      : level_(level) {
    unsigned blanks =
        (unsigned)(level > kLOG_DEBUG ? (level - kLOG_DEBUG) * 4 : 1);
    printer_ << logs[level] << " " << info << ":" << std::string(blanks, ' ');
  }

  template <typename T> ToLog &operator<<(T const &value) {
    printer_ << value;
    return *this;
  }

  ~ToLog() { printer_.print(level_); }

private:
  LogLevel level_;
  Print printer_;
};

#define LOG(level)                                                             \
  if (level > paddle_mobile::log_level) {                                      \
  } else                                                                       \
  paddle_mobile::ToLog(level,                                                  \
                       (std::stringstream()                                    \
                        << "[file: " << (strrchr(__FILE__, '/')                \
                                             ? (strrchr(__FILE__, '/') + 1)    \
                                             : __FILE__)                       \
                        << "] [line: " << __LINE__ << "] ")                    \
                           .str())

#define DLOG                                                                   \
  paddle_mobile::ToLog(paddle_mobile::kLOG_DEBUG,                              \
                       (std::stringstream()                                    \
                        << "[file: " << (strrchr(__FILE__, '/')                \
                                             ? (strrchr(__FILE__, '/') + 1)    \
                                             : __FILE__)                       \
                        << "] [line: " << __LINE__ << "] ")                    \
                           .str())
}

#else

namespace paddle_mobile {

enum LogLevel {
  kNO_LOG,
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4
};

struct ToLog;
struct Print {
  friend struct ToLog;
  template <typename T> Print &operator<<(T const &value) {}

private:
};

struct ToLog {
  ToLog(LogLevel level) {}

  template <typename T> ToLog &operator<<(T const &value) { return *this; }
};

#define LOG(level)                                                             \
  if (true) {                                                                  \
  } else                                                                       \
  paddle_mobile::ToLog(level)

#define DLOG                                                                   \
  if (true) {                                                                  \
  } else                                                                       \
  paddle_mobile::ToLog(paddle_mobile::kLOG_DEBUG)
}
#endif
