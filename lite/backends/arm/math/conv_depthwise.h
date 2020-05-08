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

#include <cmath>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_3x3s1_depthwise_fp32(const float* i_data,
                               float* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);

void conv_3x3s2_depthwise_fp32(const float* i_data,
                               float* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);

void conv_depthwise_3x3s1_fp32(const float* din,
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
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);

void conv_depthwise_3x3s2_fp32(const float* din,
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
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s1_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s2_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx);

void conv_depthwise_5x5s1_fp32(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               const operators::ConvParam& param,
                               ARMContext* ctx);

void conv_depthwise_5x5s2_fp32(const float* din,
                               float* dout,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const float* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);

void conv_depthwise_5x5s2p2_fp32(const float* din,
                                 float* dout,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win,
                                 const float* weights,
                                 const float* bias,
                                 int pad,
                                 bool flag_bias,
                                 bool flag_relu,
                                 ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_5x5s1_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_5x5s2_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx);

void conv_depthwise_3x3s1p0_bias_relu(float* dout,
                                      const float* din,
                                      const float* weights,
                                      const float* bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext* ctx);

void conv_depthwise_3x3s1p0_bias_s_relu(float* dout,
                                        const float* din,
                                        const float* weights,
                                        const float* bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext* ctx);

void conv_depthwise_3x3s1p1_bias_relu(float* dout,
                                      const float* din,
                                      const float* weights,
                                      const float* bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext* ctx);

void conv_depthwise_3x3s1p1_bias_s_relu(float* dout,
                                        const float* din,
                                        const float* weights,
                                        const float* bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext* ctx);

void conv_depthwise_3x3s2p0_bias_relu(float* dout,
                                      const float* din,
                                      const float* weights,
                                      const float* bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext* ctx);

void conv_depthwise_3x3s2p0_bias_s_relu(float* dout,
                                        const float* din,
                                        const float* weights,
                                        const float* bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext* ctx);

void conv_depthwise_3x3s2p1_bias_relu(float* dout,
                                      const float* din,
                                      const float* weights,
                                      const float* bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext* ctx);

void conv_depthwise_3x3s2p1_bias_s_relu(float* dout,
                                        const float* din,
                                        const float* weights,
                                        const float* bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
