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

#include <immintrin.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite_metal {
namespace x86 {
namespace math {

// for input and filter pack
void pack8_m256(lite_metal::Tensor* input,
                lite_metal::Tensor* output,
                const int channel_num,
                const bool is_filter);
void pack4_m128(lite_metal::Tensor* input,
                lite_metal::Tensor* output,
                const int channel_num,
                const bool is_filter);

// for output unpack
void unpack8_m256(lite_metal::Tensor* input, lite_metal::Tensor* output);
void unpack4_m128(lite_metal::Tensor* input, lite_metal::Tensor* output);

// for input padding
void padding8_m256(lite_metal::Tensor* input,
                   lite_metal::Tensor* output,
                   const std::vector<int>& paddings);
void padding4_m128(lite_metal::Tensor* input,
                   lite_metal::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite_metal::Tensor* input,
                    lite_metal::Tensor* output,
                    const std::vector<int>& paddings);

void pack_padding8_m256(lite_metal::Tensor* input,
                        lite_metal::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);

// for activation - only support relu, relu6
__m256 activation8_m256(__m256 input, const lite_metal_api::ActivationType act_type);
__m128 activation4_m128(__m128 input, const lite_metal_api::ActivationType act_type);
float activation1_float(float input, const lite_metal_api::ActivationType act_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
