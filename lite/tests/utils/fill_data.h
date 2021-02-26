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

#include <random>
template <typename Dtype>
inline void fill_data_const(Dtype* dio, Dtype value, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}

template <typename Dtype>
inline void fill_data_rand(Dtype* dio, Dtype vstart, Dtype vend, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    dio[i] = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
  }
}

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;

inline float16_t convert_half(float val) {
  // float -> uint64
  uint64_t val2 = *(uint64_t*)(&val);  // NOLINT
  // fraction = 9-31, expand = 1-8, sign = 0
  // fraction = 0x007fffff, expand = 0x7f800000, sign = 0x80000000
  // [-127, 128] -> [-31, 32], 127 - 15 = 112(left 10)
  uint16_t t = ((val2 & 0x007fffff) >> 13) | ((val2 & 0x80000000) >> 16) |
               (((val2 & 0x7f800000) >> 13) - (112 << 10));
  // round remind
  if (val2 & 0x1000) {
    t++;
  }
  float16_t h = *(float16_t*)(&t);  // NOLINT
  return h;
}

inline float convert_full(float16_t val) {
  uint16_t val2 = *(uint16_t*)(&val);  // NOLINT
  uint16_t frac = (val2 & 0x3ff) | 0x400;
  int exp = ((val2 & 0x7c00) >> 10) - 25;
  float m;
  if (frac == 0 && exp == 0x1f) {
    m = 0x7F800000;
  } else if (frac || exp) {
    m = frac * pow(2, exp);
  } else {
    m = 0.f;
  }
  return (val2 & 0x8000) ? -m : m;
}

inline void float_to_fp16(const float* src, float16_t* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (std::abs(src[i]) <= 5e-4f) {
      dst[i] = src[i];
    } else {
      dst[i] = convert_half(src[i]);
    }
  }
}

inline void fp16_to_float(const float16_t* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (std::abs(src[i]) <= 5e-4f) {
      dst[i] = src[i];
    } else {
      dst[i] = convert_full(src[i]);
    }
  }
}
#endif  // ENABLE_ARM_FP16
