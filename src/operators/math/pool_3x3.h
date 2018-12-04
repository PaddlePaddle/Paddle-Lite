/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef POOL_OP

#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <vector>
#include "framework/tensor.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace paddle_mobile {
namespace operators {
namespace math {
void Pool3x3Avgs1p1(const framework::Tensor *input, framework::Tensor *output);
void Pool3x3Maxs1p1(const framework::Tensor *input, framework::Tensor *output);
void Pool3x3Max(std::vector<int> strides, std::vector<int> paddings,
                const framework::Tensor *input, framework::Tensor *output);

void Pool3x3Avg(std::vector<int> strides, std::vector<int> paddings,
                const framework::Tensor *in_x, framework::Tensor *out);

void Pool3x3Maxs1_int8(const framework::Tensor *input,
                       framework::Tensor *output, int32_t pad_h, int32_t pad_w);
void Pool3x3Maxs2_int8(const framework::Tensor *input,
                       framework::Tensor *output, int32_t pad_h, int32_t pad_w);
void Pool3x3Max_int8(const std::vector<int> &strides,
                     const std::vector<int> &paddings,
                     const framework::Tensor *input, framework::Tensor *output);
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
