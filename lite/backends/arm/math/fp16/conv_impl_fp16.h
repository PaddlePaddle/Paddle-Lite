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

#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;
void conv1x1s1_gemm_fp16(const float16_t* din,
                         float16_t* dout,
                         int num,
                         int chout,
                         int hout,
                         int wout,
                         int chin,
                         int hin,
                         int win,
                         const float16_t* weights,
                         const float16_t* bias,
                         const operators::ConvParam& param,
                         ARMContext* ctx);

void conv_im2col_gemm_fp16(const float16_t* din,
                           float16_t* dout,
                           int num,
                           int chout,
                           int hout,
                           int wout,
                           int chin,
                           int hin,
                           int win,
                           const float16_t* weights,
                           const float16_t* bias,
                           const operators::ConvParam& param,
                           ARMContext* ctx);

size_t conv3x3s2_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx);

void conv_3x3s2_direct_fp16(const float16_t* i_data,
                            float16_t* o_data,
                            int bs,
                            int oc,
                            int oh,
                            int ow,
                            int ic,
                            int ih,
                            int win,
                            const float16_t* weights,
                            const float16_t* bias,
                            const operators::ConvParam& param,
                            ARMContext* ctx);

void im2col_fp16(const float16_t* data_im,
                 int channels,
                 int height,
                 int width,
                 int kernel_h,
                 int kernel_w,
                 int pad_top,
                 int pad_bottom,
                 int pad_left,
                 int pad_right,
                 int stride_h,
                 int stride_w,
                 int dilation_h,
                 int dilation_w,
                 float16_t* data_col);

void im2col_common_fp16(const float16_t* data_im,
                        int channels,
                        int height,
                        int width,
                        int kernel_h,
                        int kernel_w,
                        int pad_top,
                        int pad_bottom,
                        int pad_left,
                        int pad_right,
                        int stride_h,
                        int stride_w,
                        int dilation_h,
                        int dilation_w,
                        float16_t* data_col);
void im2col_s1_fp16(const float16_t* data_im,
                    int channels,
                    int height,
                    int width,
                    int kernel_h,
                    int kernel_w,
                    int pad_top,
                    int pad_bottom,
                    int pad_left,
                    int pad_right,
                    int dilation_h,
                    int dilation_w,
                    float16_t* data_col);

void im2col_s2_fp16(const float16_t* data_im,
                    int channels,
                    int height,
                    int width,
                    int kernel_h,
                    int kernel_w,
                    int pad_top,
                    int pad_bottom,
                    int pad_left,
                    int pad_right,
                    int dilation_h,
                    int dilation_w,
                    float16_t* data_col);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
