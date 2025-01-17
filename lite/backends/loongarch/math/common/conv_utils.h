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

#include "lite/backends/loongarch/xxl.h"
#include <vector>
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace loongarch {
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

#if __loongarch_asx
// for input padding
void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
#endif
void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings);
#if __loongarch_asx
void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);
#endif

// for activation - only support relu, relu6, leakyRelu, hard_swish
#ifdef __loongarch_asx
__m256 activation8_m256(__m256 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
#endif
__m128 activation4_m128(__m128 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
float activation1_float(float input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
#if __loongarch_asx
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
#endif

#if __loongarch_sx
void packC4_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel);

void unpackC4_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel);
#endif

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

#ifdef __loongarch_asx
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
  __t0 = lasx_unpacklo_f32(row0, row1);
  __t1 = lasx_unpackhi_f32(row0, row1);
  __t2 = lasx_unpacklo_f32(row2, row3);
  __t3 = lasx_unpackhi_f32(row2, row3);
  __t4 = lasx_unpacklo_f32(row4, row5);
  __t5 = lasx_unpackhi_f32(row4, row5);
  __t6 = lasx_unpacklo_f32(row6, row7);
  __t7 = lasx_unpackhi_f32(row6, row7);
  __tt0 = lasx_shuffle_f32(__t0, __t2, LSX_SHUFFLE(1, 0, 1, 0));
  __tt1 = lasx_shuffle_f32(__t0, __t2, LSX_SHUFFLE(3, 2, 3, 2));
  __tt2 = lasx_shuffle_f32(__t1, __t3, LSX_SHUFFLE(1, 0, 1, 0));
  __tt3 = lasx_shuffle_f32(__t1, __t3, LSX_SHUFFLE(3, 2, 3, 2));
  __tt4 = lasx_shuffle_f32(__t4, __t6, LSX_SHUFFLE(1, 0, 1, 0));
  __tt5 = lasx_shuffle_f32(__t4, __t6, LSX_SHUFFLE(3, 2, 3, 2));
  __tt6 = lasx_shuffle_f32(__t5, __t7, LSX_SHUFFLE(1, 0, 1, 0));
  __tt7 = lasx_shuffle_f32(__t5, __t7, LSX_SHUFFLE(3, 2, 3, 2));
  row0 = lasx_permute2f128_f32(__tt0, __tt4, 0x20);
  row1 = lasx_permute2f128_f32(__tt1, __tt5, 0x20);
  row2 = lasx_permute2f128_f32(__tt2, __tt6, 0x20);
  row3 = lasx_permute2f128_f32(__tt3, __tt7, 0x20);
  row4 = lasx_permute2f128_f32(__tt0, __tt4, 0x31);
  row5 = lasx_permute2f128_f32(__tt1, __tt5, 0x31);
  row6 = lasx_permute2f128_f32(__tt2, __tt6, 0x31);
  row7 = lasx_permute2f128_f32(__tt3, __tt7, 0x31);
}
#endif

#if __loongarch_sx
inline void transpose4_ps(__m128& row0,
                          __m128& row1,
                          __m128& row2,
                          __m128& row3) {
  __m128 tmp3, tmp2, tmp1, tmp0;
  tmp0 = lsx_unpacklo_f32((row0), (row1));
  tmp2 = lsx_unpacklo_f32((row2), (row3));
  tmp1 = lsx_unpackhi_f32((row0), (row1));
  tmp3 = lsx_unpackhi_f32((row2), (row3));
  row0 = lsx_movelh_f32(tmp0, tmp2);
  row1 = lsx_movehl_f32(tmp2, tmp0);
  row2 = lsx_movelh_f32(tmp1, tmp3);
  row3 = lsx_movehl_f32(tmp3, tmp1);
}
#endif
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
