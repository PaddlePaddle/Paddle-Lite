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

#include "lite/core/tensor.h"

namespace paddle {
namespace lite_metal {
namespace x86 {
namespace math {

void conv_depthwise_m128(lite_metal::Tensor* input,
                         lite_metal::Tensor* output,
                         lite_metal::Tensor* filter,
                         lite_metal::Tensor* bias,
                         const int stride_h,
                         const int stride_w,
                         const int dilation_h,
                         const int dilation_w,
                         const bool has_act,
                         const lite_metal_api::ActivationType act_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
