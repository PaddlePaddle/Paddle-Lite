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

/// conv3x3s2
size_t conv3x3s2_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx);
void conv_3x3s2_direct_fp32(const float* din,
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
                            ARMContext* ctx);

template <typename Dtype>
void conv_3x3s2_direct_int8(const int8_t* din,
                            Dtype* dout,
                            int num,
                            int chout,
                            int hout,
                            int wout,
                            int chin,
                            int hin,
                            int win,
                            const int8_t* weights,
                            const float* bias,
                            const operators::ConvParam& param,
                            ARMContext* ctx,
                            const float* scale);

void conv1x1s1_gemm(const float* din,
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
                    ARMContext* ctx);

template <typename Dtype>
void conv1x1s1_gemm_int8(const int8_t* din,
                         Dtype* dout,
                         int num,
                         int chout,
                         int hout,
                         int wout,
                         int chin,
                         int hin,
                         int win,
                         const int8_t* weights,
                         const float* bias,
                         const operators::ConvParam& param,
                         ARMContext* ctx,
                         const float* scale);

void conv_im2col_gemm(const float* din,
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
                      ARMContext* ctx);

template <typename Dtype>
void conv_im2col_gemm_int8(const int8_t* din,
                           Dtype* dout,
                           int num,
                           int chout,
                           int hout,
                           int wout,
                           int chin,
                           int hin,
                           int win,
                           const int8_t* weights,
                           const float* bias,
                           const operators::ConvParam& param,
                           ARMContext* ctx,
                           const float* scale);

/// depthwise conv
void conv_depthwise_3x3_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale);

void conv_depthwise_3x3_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale);

void conv_depthwise_3x3_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale);

void conv_depthwise_5x5_fp32(const void* din,
                             void* dout,
                             int num,
                             int ch_out,
                             int h_out,
                             int w_out,
                             int ch_in,
                             int h_in,
                             int w_in,
                             const void* weights,
                             const float* bias,
                             const operators::ConvParam& param,
                             ARMContext* ctx,
                             const float* scale);

void conv_depthwise_5x5_int8_fp32(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale);

void conv_depthwise_5x5_int8_int8(const void* din,
                                  void* dout,
                                  int num,
                                  int ch_out,
                                  int h_out,
                                  int w_out,
                                  int ch_in,
                                  int h_in,
                                  int w_in,
                                  const void* weights,
                                  const float* bias,
                                  const operators::ConvParam& param,
                                  ARMContext* ctx,
                                  const float* scale);

/// winograd conv, only support 3x3s1
void conv_winograd3x3(const float* din,
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
                      ARMContext* ctx);

void winograd_transform_weights(
    void* dout, const void* din, int ch_out, int ch_in, void* work_space);

// new winograd
void weight_trans_c4_8x8(
    float* dest, const float* src, int ic, int oc, void* workspace);
void weight_trans_c4_6x6(
    float* dest, const float* src, int ic, int oc, void* workspace);
void weight_trans_c4_4x4(
    float* dest, const float* src, int ic, int oc, void* workspace);
void conv_compute_6x6_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx);
void conv_compute_4x4_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx);
void conv_compute_2x2_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx);
void conv_compute_2x2_3x3_small(const float* input,
                                float* output,
                                int num,
                                int chout,
                                int hout,
                                int wout,
                                int chin,
                                int hin,
                                int win,
                                const float* weight,
                                const float* bias,
                                const operators::ConvParam& param,
                                ARMContext* ctx);
void input_trans_c8_4x4_int8(const int8_t* src,
                             int src_stride,
                             int src_h_stride,
                             int16_t* dest,
                             int dest_stride,
                             int dest_h_stride);
void output_trans_c8_post_2x4_int8(const int32_t* src,
                                   int src_stride,
                                   int src_h_stride,
                                   int32_t* dest,
                                   int dest_stride,
                                   int dest_h_stride);
void weight_trans_c8_4x4_int8(
    int16_t* dest, const int8_t* src, int ic, int oc, void* workspace);
void weight_trans_c8_6x6_int8(
    int16_t* dest, const int8_t* src, int ic, int oc, void* workspace);
template <typename Dtype>
void conv_compute_2x2_3x3_int8(const int8_t* input,
                               Dtype* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const int16_t* weight,
                               const float* bias,
                               const float* scale,
                               const operators::ConvParam& param,
                               ARMContext* ctx);
template <typename Dtype>
void conv_compute_4x4_3x3_int8(const int8_t* input,
                               Dtype* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const int16_t* weight,
                               const float* bias,
                               const float* scale,
                               const operators::ConvParam& param,
                               ARMContext* ctx);

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
}  // namespace arm
}  // namespace lite
}  // namespace paddle
