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

#include <gflags/gflags.h>
#if !defined(_WIN32)
#include <sys/time.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#include "lite/backends/x86/port.h"
#endif
#include <time.h>
#include <cmath>

// for eval
DEFINE_string(model_dir, "", "model dir");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_int32(power_mode,
             3,
             "arm power mode: "
             "0 for big cluster, "
             "1 for little cluster, "
             "2 for all cores, "
             "3 for no bind");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(im_width, 224, "image width");
DEFINE_int32(im_height, 224, "image height");
DEFINE_bool(int8, false, "is run int8");

namespace paddle {
namespace lite {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

}  // namespace lite
}  // namespace paddle
