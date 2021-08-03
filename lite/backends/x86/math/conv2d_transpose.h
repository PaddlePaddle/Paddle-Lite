/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void col2im(const float* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h0,
            const int pad_h1,
            const int pad_w0,
            const int pad_w1,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            float* data_im);

void fill_bias_act(float* tensor,
                   const float* bias,
                   int channel,
                   int channel_size,
                   bool flag_bias,
                   const operators::ActivationParam* act_param);

void conv_transpose_depthwise_s1(const float* dst,
                                 const float* weights,
                                 const int channels,
                                 const int height,
                                 const int width,
                                 const int kernel_h,
                                 const int kernel_w,
                                 const int pad_h0,
                                 const int pad_h1,
                                 const int pad_w0,
                                 const int pad_w1,
                                 const int dilation_h,
                                 const int dilation_w,
                                 float* src,
                                 X86Context* ctx);

void conv_transpose_depthwise_s2(const float* dst,
                                 const float* weights,
                                 const int channels,
                                 const int height,
                                 const int width,
                                 const int kernel_h,
                                 const int kernel_w,
                                 const int pad_h0,
                                 const int pad_h1,
                                 const int pad_w0,
                                 const int pad_w1,
                                 const int dilation_h,
                                 const int dilation_w,
                                 float* src,
                                 X86Context* ctx);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
