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
inline void data_diff(const float16_t* src1_truth,
                      const float16_t* src2,
                      float16_t* dst,
                      int size,
                      double& max_ratio,   // NOLINT
                      double& max_diff) {  // NOLINT
  const double eps = 1e-6f;
  max_diff = fabs(src1_truth[0] - src2[0]);
  dst[0] = max_diff;
  max_ratio = fabs(max_diff) / (std::abs(src1_truth[0]) + eps);
  for (int i = 1; i < size; ++i) {
    double diff = fabs(src1_truth[i] - src2[i]);
    dst[i] = diff;
    max_diff = max_diff < diff ? diff : max_diff;
    double ratio = fabs(diff) / (std::abs(src1_truth[i]) + eps);
    if (max_ratio < ratio) {
      max_ratio = ratio;
    }
  }
}

inline float16_t convert_half(float val) {
  uint64_t val2 = *(uint64_t*)(&val);  // NOLINT
  uint16_t t = ((val2 & 0x007fffff) >> 13) | ((val2 & 0x80000000) >> 16) |
               (((val2 & 0x7f800000) >> 13) - (112 << 10));
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
    dst[i] = convert_half(src[i]);
  }
}

inline void fp16_to_float(const float16_t* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = convert_full(src[i]);
  }
}

inline void print_tensor(const float16_t* din, int64_t size, int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%.6f ", convert_full(din[i]));
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}
#endif  // ENABLE_ARM_FP16
