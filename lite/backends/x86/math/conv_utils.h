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

#include <immintrin.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// for input and filter pack
void pack8_m256(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);
void pack4_m128(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);

// for output unpack
void unpack8_m256(lite::Tensor* input, lite::Tensor* output);
void unpack4_m128(lite::Tensor* input, lite::Tensor* output);

// for input padding
void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings);

void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);
void pack_with_pad(const float* din, float* dout, int pad, int h_in, int w_in, int real_hout_c_block);

void packC8_with_Cleft(const float* din, float* dout, const std::vector<int>& pad, int h_in, int w_in, int channel);

void unpack_output(const float* din, float * dout, int size_out_channel, int real_hout_c_block);

void unpackC8_with_Cleft(const float* din, float* dout, int size_out_channel, int channel);

// for activation - only support relu, relu6
__m256 activation8_m256(__m256 input, const lite_api::ActivationType act_type);
__m128 activation4_m128(__m128 input, const lite_api::ActivationType act_type);
float activation1_float(float input, const lite_api::ActivationType act_type);
template <typename Dtype>
void im2col(const Dtype* data_im,
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
            Dtype* data_col);

template <typename Dtype>
void im2col_common(const Dtype* data_im,
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
                   Dtype* data_col);

template <typename Dtype>
void im2col_s1(const Dtype* data_im,
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
               Dtype* data_col);

template <typename Dtype>
void im2col_s2(const Dtype* data_im,
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
               Dtype* data_col);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
