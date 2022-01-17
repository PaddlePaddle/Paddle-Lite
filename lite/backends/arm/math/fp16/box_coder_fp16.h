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

#include <string>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;
void encode_bbox_center_kernel(const int batch_num,
                               const float16_t* loc_data,
                               const float16_t* prior_data,
                               const float16_t* variance,
                               const bool var_len4,
                               const bool normalized,
                               const int num_priors,
                               float16_t* bbox_data);

void decode_bbox_center_kernel(const int batch_num,
                               const float16_t* loc_data,
                               const float16_t* prior_data,
                               const float16_t* variance,
                               const bool var_len4,
                               const int num_priors,
                               const bool normalized,
                               float16_t* bbox_data);

void decode_center_size_axis_1(const int var_size,
                               const int row,
                               const int col,
                               const int len,
                               const float16_t* target_box_data,
                               const float16_t* prior_box_data,
                               const float16_t* prior_box_var_data,
                               const bool normalized,
                               const std::vector<float16_t> variance,
                               float16_t* output);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
