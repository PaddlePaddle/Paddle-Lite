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

#include <chrono>  // NOLINT
#include <list>

namespace lite {
namespace test {

class Timer final {
 public:
  Timer() {}

  ~Timer() {}

  void clear() { ms_time_.clear(); }

  void start() { tstart_ = std::chrono::system_clock::now(); }

  void end() {
    tend_ = std::chrono::system_clock::now();
    auto ts =
        std::chrono::duration_cast<std::chrono::microseconds>(tend_ - tstart_);
    float elapse_ms = 1000.f * static_cast<float>(ts.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den;
    ms_time_.push_back(elapse_ms);
  }

  float get_average_ms() {
    if (ms_time_.size() == 0) {
      return 0.f;
    }
    float sum = 0.f;
    for (auto i : ms_time_) {
      sum += i;
    }
    return sum / ms_time_.size();
  }

  float get_sum_ms() {
    if (ms_time_.size() == 0) {
      return 0.f;
    }
    float sum = 0.f;
    for (auto i : ms_time_) {
      sum += i;
    }
    return sum;
  }

  // return tile (0-99) time.
  float get_tile_time(float tile) {
    if (tile < 0 || tile > 100) {
      return -1.f;
    }
    int total_items = static_cast<int>(ms_time_.size());
    if (total_items <= 0) {
      return -2.f;
    }
    ms_time_.sort();
    int pos = static_cast<int>(tile * total_items / 100);
    auto it = ms_time_.begin();
    for (int i = 0; i < pos; ++i) {
      ++it;
    }
    return *it;
  }

  std::list<float> get_time_stat() { return ms_time_; }

  float get_min_time() {
    ms_time_.sort();
    return *ms_time_.begin();
  }

  float get_max_time() {
    ms_time_.sort([](int a, int b) { return a > b; });
    return *ms_time_.begin();
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> tstart_;
  std::chrono::time_point<std::chrono::system_clock> tend_;
  std::list<float> ms_time_;
};

}  // namespace test
}  // namespace lite
