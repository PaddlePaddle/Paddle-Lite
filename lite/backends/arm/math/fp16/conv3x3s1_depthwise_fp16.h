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
#include <arm_neon.h>
#include <cmath>
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
// typedef __fp16 float16_t;
void conv_depthwise_3x3s1p1_bias_fp16_fp16(float16_t* dout,
                                           const float16_t* din,
                                           const float16_t* weights,
                                           const float16_t* bias,
                                           bool flag_bias,
                                           int flag_act,
                                           float* alpha,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
