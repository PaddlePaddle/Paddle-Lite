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

#include <vector>
#include "lite/backends/x86/math/math_function.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void encode_center_size(const int64_t row,
                        const int64_t col,
                        const int64_t len,
                        const float* target_box_data,
                        const float* prior_box_data,
                        const float* prior_box_var_data,
                        const bool normalized,
                        const std::vector<float> variance,
                        float* output);

void decode_center_size(const int axis,
                        const int var_size,
                        const int64_t row,
                        const int64_t col,
                        const int64_t len,
                        const float* target_box_data,
                        const float* prior_box_data,
                        const float* prior_box_var_data,
                        const bool normalized,
                        const std::vector<float> variance,
                        float* output);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
