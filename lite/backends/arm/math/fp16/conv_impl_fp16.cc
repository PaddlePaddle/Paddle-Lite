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

#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/backends/arm/math/fp16/conv3x3s1_depthwise_fp16.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
void conv_depthwise_3x3_fp16(const float16_t* din,
                             float16_t* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const float16_t* weights,
                             const float16_t* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx) {
  auto paddings = *param.paddings;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  conv_depthwise_3x3s1p1_bias_fp16_fp16(
      reinterpret_cast<float16_t*>(dout),
      reinterpret_cast<const float16_t*>(din),
      reinterpret_cast<const float16_t*>(weights),
      reinterpret_cast<const float16_t*>(bias),
      flag_bias,
      flag_act,
      alpha,
      num,
      ch_in,
      h_in,
      w_in,
      h_out,
      w_out,
      ctx);
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
