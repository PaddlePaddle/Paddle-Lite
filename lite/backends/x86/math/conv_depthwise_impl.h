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
void conv_depthwise_3x3s1_p01_direct(
    const float* din,
    float* dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float* weights,
    const float* bias,
    int pad,
    bool flag_bias,
    const operators::ActivationParam act_param);
void conv_depthwise_3x3s2_p01_direct(
    const float* din,
    float* dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float* weights,
    const float* bias,
    int pad,
    bool flag_bias,
    const operators::ActivationParam act_param);
void conv_depthwise_5x5s1(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param);
void conv_depthwise_5x5s2(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param);
void conv_depthwise_3x3_pack(const operators::ConvParam& param,
                             lite::Tensor* input_padding_,
                             lite::Tensor* input_pack_,
                             lite::Tensor* filter_pack_,
                             lite::Tensor* output_pack_);
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
