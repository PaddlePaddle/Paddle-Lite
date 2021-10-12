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
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_depthwise_3x3s1_m256(lite::Tensor* input,
                               lite::Tensor* output,
                               lite::Tensor* filter,
                               lite::Tensor* bias,
                               const bool has_act,
                               const lite_api::ActivationType act_type,
                               const operators::ActivationParam act_param);

void conv_depthwise_3x3s2_m256(lite::Tensor* input,
                               lite::Tensor* output,
                               lite::Tensor* filter,
                               lite::Tensor* bias,
                               const bool has_act,
                               const lite_api::ActivationType act_type,
                               const operators::ActivationParam act_param);

void conv_depthwise_m256(lite::Tensor* input,
                         lite::Tensor* output,
                         lite::Tensor* filter,
                         lite::Tensor* bias,
                         const int stride_h,
                         const int stride_w,
                         const int dilation_h,
                         const int dilation_w,
                         const bool has_act,
                         const lite_api::ActivationType act_type,
                         const operators::ActivationParam act_param);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
