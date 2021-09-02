// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/sparse_conv_impl.h"
#include <arm_neon.h>
#include <vector>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sparse_conv_fp32_pipelined(const float* A,
                                const float* B,
                                const int32_t* widx_dmap,
                                const uint32_t* nidx_nnzmap,
                                const float* bias,
                                float* output,
                                const int M,
                                const int K,
                                const int N,
                                const operators::SparseConvParam& param,
                                ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

void sparse_conv_int8_fp32_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     float* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

void sparse_conv_int8_int8_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     int8_t* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
