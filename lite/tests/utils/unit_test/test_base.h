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

#pragma once

#include <stdlib.h>  // for getenv in arm
#include <string.h>
#include <chrono>  // NOLINT
namespace lite {
namespace test {
class Test {
 public:
  Test();
  ~Test();

  template <typename ClassDerive>
  static ClassDerive* get_instance() {
    static ClassDerive* ins = new ClassDerive;
    return ins;
  }

 protected:
  /**
   * * \brief set test case class global initial.
   * */
  virtual void setup() {}

  /**
  * \brief tear down the test class when test over.
  */
  virtual void teardown() {}
};

Test::Test() {
  // SetUp();
}

Test::~Test() {
  // TearDown();
}

class Counter {
 public:
  /**
  *  0 = nanoseconds
  *  1 = microseconds
  *  2 = milliseconds
  *  3 = seconds
  *  4 = minutes
  *  5 = hours
  *  default milliseconds
  */
  Counter() : _precision(2) {}
  explicit Counter(int p) : _precision(p) {}
  inline void start() { _start = std::chrono::system_clock::now(); }

  inline void end() { _end = std::chrono::system_clock::now(); }

  inline double elapsed_time() {
    switch (_precision) {
      case 0:
        return std::chrono::duration_cast<std::chrono::nanoseconds>(_end -
                                                                    _start)
            .count();
      case 1:
        return std::chrono::duration_cast<std::chrono::microseconds>(_end -
                                                                     _start)
            .count();
      case 2:
        return std::chrono::duration_cast<std::chrono::milliseconds>(_end -
                                                                     _start)
            .count();
      case 3:
        return std::chrono::duration_cast<std::chrono::seconds>(_end - _start)
            .count();
      case 4:
        return std::chrono::duration_cast<std::chrono::minutes>(_end - _start)
            .count();
      case 5:
        return std::chrono::duration_cast<std::chrono::hours>(_end - _start)
            .count();
    }
    return -1;
  }

 private:
  int _precision;
  std::chrono::time_point<std::chrono::system_clock> _start;
  std::chrono::time_point<std::chrono::system_clock> _end;
};

namespace config {
static bool terminalSupportColor;
inline void initial() {
  if (const char* term = getenv("TERM")) {
    if (0 == strcmp(term, "cygwin") || 0 == strcmp(term, "linux") ||
        0 == strcmp(term, "screen") || 0 == strcmp(term, "xterm") ||
        0 == strcmp(term, "xterm-256color") ||
        0 == strcmp(term, "xterm-color")) {
      terminalSupportColor = true;
    }
  } else {
    terminalSupportColor = false;
  }
}
}  // namespace config

inline const char* red() {
  return config::terminalSupportColor ? "\e[31m" : "";
}
inline const char* b_red() {
  return config::terminalSupportColor ? "\e[41m" : "";
}
inline const char* green() {
  return config::terminalSupportColor ? "\e[32m" : "";
}
inline const char* yellow() {
  return config::terminalSupportColor ? "\e[33m" : "";
}
inline const char* blue() {
  return config::terminalSupportColor ? "\e[34m" : "";
}
inline const char* purple() {
  return config::terminalSupportColor ? "\e[35m" : "";
}
inline const char* cyan() {
  return config::terminalSupportColor ? "\e[36m" : "";
}
inline const char* light_gray() {
  return config::terminalSupportColor ? "\e[37m" : "";
}
inline const char* white() {
  return config::terminalSupportColor ? "\e[37m" : "";
}
inline const char* light_red() {
  return config::terminalSupportColor ? "\e[91m" : "";
}
inline const char* dim() { return config::terminalSupportColor ? "\e[2m" : ""; }
inline const char* bold() {
  return config::terminalSupportColor ? "\e[1m" : "";
}
inline const char* underline() {
  return config::terminalSupportColor ? "\e[4m" : "";
}
inline const char* blink() {
  return config::terminalSupportColor ? "\e[5m" : "";
}
inline const char* reset() {
  return config::terminalSupportColor ? "\e[0m" : "";
}

}  // namespace test
}  // namespace lite
