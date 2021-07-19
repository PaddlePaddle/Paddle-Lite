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
#include <chrono>  // NOLINT(build/c++11)
#include <string>

namespace paddle {
namespace lite {

using std::chrono::time_point;
using std::chrono::system_clock;

// thread-safe timer impl
class Timer {
 public:
  explicit Timer(const std::string timer_info = "benchmark");
  void Start();
  float Stop();
  void SleepInMs(const float milliseconds);
  void Print();

 private:
  void Reset();

 private:
  float min_;
  float max_;
  float sum_;
  std::string timer_info_;
  time_point<system_clock> start_;
  time_point<system_clock> stop_;
  int count_;
};

}  // namespace lite
}  // namespace paddle
