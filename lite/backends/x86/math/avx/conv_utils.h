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
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// tranpose [chout, chin, wh, ww] to [chout/block,chin,wh,ww,block]
// dout space should be allocated before calling conv_trans_weights_numc
void conv_trans_weights_numc(const float* din,
                             float* dout,  // dout has been expanded
                             int chout,
                             int chin,
                             int wh,
                             int ww,
                             int block);

// tranpose [chout,chin,wh,ww] to [chout/block,wh,ww,chin,block]
// this function is different from conv_trans_weights_numc just
// in that we make chw->hwc
void conv_trans_weights_numc_c3(const float* din,
                                float* dout,
                                int chout,
                                int chin,
                                int wh,
                                int ww,
                                int block);

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

// for activation - only support relu, relu6, leakyRelu, hard_swish
__m256 activation8_m256(__m256 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
__m128 activation4_m128(__m128 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
float activation1_float(float input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
void packC8_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel);

void unpackC8_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel);

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

// From: https://stackoverflow.com/a/25627536
inline void transpose8_ps(__m256& row0,  // NOLINT
                          __m256& row1,  // NOLINT
                          __m256& row2,  // NOLINT
                          __m256& row3,  // NOLINT
                          __m256& row4,  // NOLINT
                          __m256& row5,  // NOLINT
                          __m256& row6,  // NOLINT
                          __m256& row7   // NOLINT
                          ) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
  __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
  __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
  __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
  __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
  __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
  __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
  __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
