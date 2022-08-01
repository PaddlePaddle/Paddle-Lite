/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "cl_common.h"

__kernel void conv2d_transpose(
    __private const int global_size_dim0,  // (out_c + 3) / 4
    __private const int global_size_dim1,  // out_w
    __private const int global_size_dim2,  // out_n * out_h
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output,
    __private const int2 input_shape,
    __private const int2 output_shape,
    __private const int2 stride_shape,
    __private const int2 align_shape,
    __private const int2 padding_shape,
    __private const int2 kernel_shape,
    __private const int2 dilation_shape,
    __private const int2 kernel_prev_shape,
    __private const int kernel_size,
    __private const int input_c_blks) {
  const int out_c_blk_idx = get_global_id(0);
  const int out_w_idx = get_global_id(1);
  const int out_nh_idx = get_global_id(2);

  if (out_c_blk_idx >= global_size_dim0 || out_w_idx >= global_size_dim1 ||
      out_nh_idx >= global_size_dim2) {
    return;
  }

  int2 out_pos = (int2)(out_c_blk_idx * output_shape.x + out_w_idx, out_nh_idx);

#ifdef BIASE_CH
  CL_DTYPE4 out0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c_blk_idx, 0));
#else
  CL_DTYPE4 out0 = 0.f;
#endif

  const int out_n_idx = out_nh_idx / output_shape.y;
  const int out_h_idx = out_nh_idx % output_shape.y;

  int kernel_start_x = max(0, (out_w_idx + align_shape.x) / stride_shape.x);
  int kernel_start_y = max(0, (out_h_idx + align_shape.y) / stride_shape.y);
  int valid_kernel_width =
      kernel_shape.x - mad24(kernel_start_x, stride_shape.x, padding_shape.x) +
      out_w_idx - 1;
  int valid_kernel_height =
      kernel_shape.y - mad24(kernel_start_y, stride_shape.y, padding_shape.y) +
      out_h_idx - 1;

#ifndef IS_DEPTHWISE
  int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
#endif
  CL_DTYPE4 in0;
  CL_DTYPE4 weights0, weights1, weights2, weights3;
#ifndef IS_DEPTHWISE
  for (int ic = 0; ic < input_c_blks; ic++) {
    int kernel_y_base = mul24(ic, kernel_prev_shape.x * kernel_prev_shape.y);
    int in_idx = mul24(ic, input_shape.x);
    kernel_x_0 = out_c_blk_idx << 2;
    kernel_x_1 = kernel_x_0 + 1;
    kernel_x_2 = kernel_x_0 + 2;
    kernel_x_3 = kernel_x_0 + 3;
#else
  int in_idx = mul24(out_c_blk_idx, input_shape.x);
#endif
    for (int k_y = valid_kernel_height, idx_h = kernel_start_y; k_y >= 0;
         k_y -= stride_shape.y, idx_h++) {
      int in_y_idx = mad24(
          out_n_idx, input_shape.y, idx_h);  // height idx of input image2d
      int in_nh_value =
          select(in_y_idx, -1, idx_h < 0 || idx_h >= input_shape.y);
      int in_width0 = kernel_start_x;
      for (int k_x = valid_kernel_width; k_x >= 0; k_x -= stride_shape.x) {
#ifndef IS_DEPTHWISE
        if (k_x % dilation_shape.x == 0 && k_y % dilation_shape.y == 0) {
          kernel_y = mad24(k_y / dilation_shape.y,
                           kernel_prev_shape.x,
                           k_x / dilation_shape.x + kernel_y_base);
          weights0 = READ_IMG_TYPE(
              CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_0, kernel_y));
          weights1 = READ_IMG_TYPE(
              CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_1, kernel_y));
          weights2 = READ_IMG_TYPE(
              CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_2, kernel_y));
          weights3 = READ_IMG_TYPE(
              CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_3, kernel_y));
        } else {
          weights0 = (CL_DTYPE4)(0.0f);
          weights1 = (CL_DTYPE4)(0.0f);
          weights2 = (CL_DTYPE4)(0.0f);
          weights3 = (CL_DTYPE4)(0.0f);
        }
#else
      if (k_x % dilation_shape.x == 0 && k_y % dilation_shape.y == 0) {
        int kernel_x =
            mad24(out_c_blk_idx, kernel_prev_shape.x, k_x / dilation_shape.x);
        weights0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                 filter,
                                 SAMPLER,
                                 (int2)(kernel_x, k_y / dilation_shape.y));
      } else {
        weights0 = (CL_DTYPE4)(0.0f);
      }
#endif
        int in_width_value0 = in_width0;
        in_width_value0 =
            select(in_idx + in_width_value0,
                   -1,
                   (in_width_value0 < 0 || in_width_value0 >= input_shape.x));
        in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value0, in_nh_value));
#ifndef IS_DEPTHWISE
        out0.x = mad(in0.x, weights0.x, out0.x);
        out0.x = mad(in0.y, weights0.y, out0.x);
        out0.x = mad(in0.z, weights0.z, out0.x);
        out0.x = mad(in0.w, weights0.w, out0.x);

        out0.y = mad(in0.x, weights1.x, out0.y);
        out0.y = mad(in0.y, weights1.y, out0.y);
        out0.y = mad(in0.z, weights1.z, out0.y);
        out0.y = mad(in0.w, weights1.w, out0.y);

        out0.z = mad(in0.x, weights2.x, out0.z);
        out0.z = mad(in0.y, weights2.y, out0.z);
        out0.z = mad(in0.z, weights2.z, out0.z);
        out0.z = mad(in0.w, weights2.w, out0.z);

        out0.w = mad(in0.x, weights3.x, out0.w);
        out0.w = mad(in0.y, weights3.y, out0.w);
        out0.w = mad(in0.z, weights3.z, out0.w);
        out0.w = mad(in0.w, weights3.w, out0.w);
#else
      out0 = mad(in0, weights0, out0);
#endif
        in_width0++;
      }
    }
#ifndef IS_DEPTHWISE
  }
#endif

  out0 = activation_type4(out0, 0.f);

#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, out0);
}

#define COMPUTE_OUTPUT(o, in0)                                                          \
  if (o == 0) {                                                                         \
    out0.x = (remain == 0) ? mad(in0, weights0.x, out0.x) : out0.x;                     \
    out0.x = (remain == 1) ? mad(in0, weights0.y, out0.x) : out0.x;                     \
    out0.x = (remain == 2) ? mad(in0, weights0.z, out0.x) : out0.x;                     \
    out0.x = (remain == 3) ? mad(in0, weights0.w, out0.x) : out0.x;                     \
  } else if (o == 1) {                                                                  \
    out0.y = (remain == 0) ? mad(in0, weights0.x, out0.y) : out0.y;                     \
    out0.y = (remain == 1) ? mad(in0, weights0.y, out0.y) : out0.y;                     \
    out0.y = (remain == 2) ? mad(in0, weights0.z, out0.y) : out0.y;                     \
    out0.y = (remain == 3) ? mad(in0, weights0.w, out0.y) : out0.y;                     \
  } else if (o == 2) {                                                                  \
    out0.z = (remain == 0) ? mad(in0, weights0.x, out0.z) : out0.z;                     \
    out0.z =                                                         \   
            (remain == 1)                                                               \
                                                                         ? mad(in0,     \
                                                                               weights0 \
                                                                                   .y,  \
                                                                               out0.z)  \
                                                                         : out0.z;      \
    out0.z = (remain == 2) ? mad(in0, weights0.z, out0.z) : out0.z;                     \
    out0.z = (remain == 3) ? mad(in0, weights0.w, out0.z) : out0.z;                     \
  } else if (o == 3) {                                                                  \
    out0.w = (remain == 0) ? mad(in0, weights0.x, out0.w) : out0.w;                     \
    out0.w = (remain == 1) ? mad(in0, weights0.y, out0.w) : out0.w;                     \
    out0.w = (remain == 2) ? mad(in0, weights0.z, out0.w) : out0.w;                     \
    out0.w = (remain == 3) ? mad(in0, weights0.w, out0.w) : out0.w;                     \
  }

__kernel void group_conv2d_transpose(
    __private const int global_size_dim0,  // (out_c + 3) / 4
    __private const int global_size_dim1,  // out_w
    __private const int global_size_dim2,  // out_n * out_h
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output,
    __private const int2 input_shape,
    __private const int2 output_shape,
    __private const int2 stride_shape,
    __private const int2 align_shape,
    __private const int2 padding_shape,
    __private const int2 kernel_shape,
    __private const int2 dilation_shape,
    __private const int2 kernel_prev_shape,
    __private const int kernel_size,
    __private const int input_c_blks,
    __private const int in_channels_per_group,
    __private const int out_channels_per_group) {
  const int out_c_blk_idx = get_global_id(0);
  const int out_w_idx = get_global_id(1);
  const int out_nh_idx = get_global_id(2);

  if (out_c_blk_idx >= global_size_dim0 || out_w_idx >= global_size_dim1 ||
      out_nh_idx >= global_size_dim2) {
    return;
  }

  int2 out_pos = (int2)(out_c_blk_idx * output_shape.x + out_w_idx, out_nh_idx);

#ifdef BIASE_CH
  CL_DTYPE4 out0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c_blk_idx, 0));
#else
  CL_DTYPE4 out0 = 0.f;
#endif

  const int out_n_idx = out_nh_idx / output_shape.y;
  const int out_h_idx = out_nh_idx % output_shape.y;

  int kernel_start_x = max(0, (out_w_idx + align_shape.x) / stride_shape.x);
  int kernel_start_y = max(0, (out_h_idx + align_shape.y) / stride_shape.y);
  int valid_kernel_width =
      kernel_shape.x - mad24(kernel_start_x, stride_shape.x, padding_shape.x) +
      out_w_idx - 1;
  int valid_kernel_height =
      kernel_shape.y - mad24(kernel_start_y, stride_shape.y, padding_shape.y) +
      out_h_idx - 1;

  CL_DTYPE4 in0;
  CL_DTYPE4 weights0;
  for (int o = 0; o < 4; ++o) {
    int group_id = (out_c_blk_idx * 4 + o) / out_channels_per_group;
    int remain =
        (out_c_blk_idx * 4 + o - group_id * out_channels_per_group) % 4;
    for (int ic = group_id * in_channels_per_group;
         ic < (group_id + 1) * in_channels_per_group;
         ++ic) {
      int in_idx = mul24(ic / 4, input_shape.x);
      for (int k_y = valid_kernel_height, idx_h = kernel_start_y; k_y >= 0;
           k_y -= stride_shape.y, idx_h++) {
        int in_y_idx = mad24(
            out_n_idx, input_shape.y, idx_h);  // height idx of input image2d
        int in_nh_value =
            select(in_y_idx, -1, idx_h < 0 || idx_h >= input_shape.y);
        int in_width0 = kernel_start_x;
        for (int k_x = valid_kernel_width; k_x >= 0; k_x -= stride_shape.x) {
          int in_width_value0 = in_width0;
          in_width_value0 =
              select(in_idx + in_width_value0,
                     -1,
                     (in_width_value0 < 0 || in_width_value0 >= input_shape.x));
          in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                              input,
                              SAMPLER,
                              (int2)(in_width_value0, in_nh_value));
          if (k_x % dilation_shape.x == 0 && k_y % dilation_shape.y == 0) {
            int kernel_y_0 = ic * kernel_prev_shape.y + k_y / dilation_shape.y;
            int kernel_x_0 =
                (((out_c_blk_idx * 4 + o) % out_channels_per_group) / 4) *
                    kernel_prev_shape.x +
                k_x / dilation_shape.x;
            weights0 = READ_IMG_TYPE(
                CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_0, kernel_y_0));
            if (ic % 4 == 0) {
              COMPUTE_OUTPUT(o, in0.x)
            } else if (ic % 4 == 1) {
              COMPUTE_OUTPUT(o, in0.y)
            } else if (ic % 4 == 2) {
              COMPUTE_OUTPUT(o, in0.z)
            } else if (ic % 4 == 3) {
              COMPUTE_OUTPUT(o, in0.w)
            }
          } else {
            weights0 = (CL_DTYPE4)(0.0f);
          }
          in_width0++;
        }
      }
    }
  }
  out0 = activation_type4(out0, 0.f);

#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, out0);
}
