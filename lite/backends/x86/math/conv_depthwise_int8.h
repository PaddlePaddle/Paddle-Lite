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

#pragma once
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <typename Dtype>
void conv_3x3s1_dw_int8(Dtype* dout,
                        const int8_t* din,
                        const int8_t* weights,
                        const float* bias,
                        int num,
                        int chin,
                        int hin,
                        int win,
                        int hout,
                        int wout,
                        int pad_h,
                        int pad_w,
                        int flag_act,
                        float alpha,
                        const float* scale,
                        X86Context* ctx);

template <typename Dtype>
void conv_3x3s2p0_dw_int8(Dtype* dout,
                          const int8_t* din,
                          const int8_t* weights,
                          const float* bias,
                          int num,
                          int chin,
                          int hin,
                          int win,
                          int hout,
                          int wout,
                          int pad_h,
                          int pad_w,
                          int flag_act,
                          float alpha,
                          const float* scale,
                          X86Context* ctx);

template <typename Dtype>
void conv_3x3s2p1_dw_int8(Dtype* dout,
                          const int8_t* din,
                          const int8_t* weights,
                          const float* bias,
                          int num,
                          int chin,
                          int hin,
                          int win,
                          int hout,
                          int wout,
                          int pad_h,
                          int pad_w,
                          int flag_act,
                          float alpha,
                          const float* scale,
                          X86Context* ctx);
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
