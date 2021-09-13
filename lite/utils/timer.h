// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <float.h>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

// thread-safe timer impl
class Timer {
 public:
  explicit Timer(const std::string timer_info = "") {
    timer_info_ = timer_info;
    Reset();
  }

  void Start() { start_ = std::chrono::system_clock::now(); }

  float Stop() {
    stop_ = std::chrono::system_clock::now();
    float ms_delta =
        std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_)
            .count() /
        1000.0f;
    min_ = static_cast<float>(fmin(min_, ms_delta));
    max_ = static_cast<float>(fmax(max_, ms_delta));
    sum_ += ms_delta;
    count_++;
    return ms_delta;
  }

  static uint64_t GetCurrentUS() {
    uint64_t usec = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    return usec;
  }

  static void SleepInMs(const float ms) {
    if (ms <= 0.f) {
      return;
    }

    const int64_t ms_int = static_cast<int64_t>(ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_int));
  }

  void Print() {
    char min_str[16];
    snprintf(min_str, sizeof(min_str), "%6.3f", min_);
    char max_str[16];
    snprintf(max_str, sizeof(max_str), "%6.3f", max_);
    char avg_str[16];
    snprintf(
        avg_str, sizeof(avg_str), "%6.3f", sum_ / static_cast<float>(count_));
    LOG(INFO) << string_format(
        "%s time cost: min = %8s ms | max = %8s ms | avg = %8s ms",
        timer_info_.c_str(),
        min_str,
        max_str,
        avg_str);
  }

 private:
  void Reset() {
    min_ = FLT_MAX;
    max_ = FLT_MIN;
    sum_ = 0.0f;
    count_ = 0;
    stop_ = start_ = std::chrono::system_clock::now();
  }

 private:
  float min_;
  float max_;
  float sum_;
  std::string timer_info_;
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> stop_;
  int count_;
};

}  // namespace lite
}  // namespace paddle
