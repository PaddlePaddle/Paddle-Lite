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
