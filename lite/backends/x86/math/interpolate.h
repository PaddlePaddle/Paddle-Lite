// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

inline float CubicConvolution1(float x, float A) {
  return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
}

inline float CubicConvolution2(float x, float A) {
  return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
}

inline void get_cubic_upsample_coefficients(float coeffs[4], float t) {
  float A = -0.75f;

  float x1 = t;
  coeffs[0] = CubicConvolution2(x1 + 1.0f, A);
  coeffs[1] = CubicConvolution1(x1, A);

  // opposite coefficients
  float x2 = 1.0f - t;
  coeffs[2] = CubicConvolution1(x2, A);
  coeffs[3] = CubicConvolution2(x2 + 1.0f, A);
}

inline float cubic_interp(
    float x0, float x1, float x2, float x3, float t) {
  float coeffs[4];
  get_cubic_upsample_coefficients(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

void bilinear_interp(const float* input_data,
                     float* output_data,
                     const float ratio_h,
                     const float ratio_w,
                     const int in_h,
                     const int in_w,
                     const int n,
                     const int c,
                     const int out_h,
                     const int out_w,
                     const bool align_corners,
                     const bool align_mode);

void nearest_interp(const float* input_data,
                    float* output_data,
                    const float ratio_h,
                    const float ratio_w,
                    const int n,
                    const int c,
                    const int in_h,
                    const int in_w,
                    const int out_h,
                    const int out_w,
                    const bool align_corners);

void bicubic_interp(const float* input_data,
                    float* output_data,
                    const float ratio_h,
                    const float ratio_w,
                    const int n,
                    const int c,
                    const int in_h,
                    const int in_w,
                    const int out_h,
                    const int out_w,
                    const bool align_corners);

void interpolate(lite::Tensor* input,
                 lite::Tensor* out_size,
                 std::vector<const lite::Tensor*> list_new_size_tensor,
                 lite::Tensor* scale_tensor,
                 lite::Tensor* output,
                 float scale,
                 std::vector<float> scale_v,
                 int out_h,
                 int out_w,
                 const int align_mode,
                 const bool align_corners,
                 const std::string interpolate_type);

void interpolate_v2(lite::Tensor* input,
                    lite::Tensor* out_size,
                    std::vector<const lite::Tensor*> list_new_size_tensor,
                    lite::Tensor* scale_tensor,
                    lite::Tensor* output,
                    float scale,
                    std::vector<float> scale_v,
                    int out_h,
                    int out_w,
                    const int align_mode,
                    const bool align_corners,
                    const std::string interpolate_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
