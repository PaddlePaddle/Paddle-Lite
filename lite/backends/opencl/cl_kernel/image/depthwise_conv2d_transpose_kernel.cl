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

__kernel void depthwise_conv2d_transpose(
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
    __private const int kernel_size,
    __private const int input_c_blks) {
  const int out_c_blk_idx = get_global_id(0);
  const int out_w_idx = get_global_id(1);
  const int out_nh_idx = get_global_id(2);

  if (out_c_blk_idx >= global_size_dim0 || out_w_idx >= global_size_dim1 ||
      out_nh_idx >= global_size_dim2) {
    return;
  }

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

  int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
  CL_DTYPE4 in0;
  CL_DTYPE4 weights0, weights1, weights2, weights3;
  int ic = out_c_blk_idx;

#ifdef BIASE_CH
  CL_DTYPE4 out0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c_blk_idx, 0));
#else
  CL_DTYPE4 out0 = 0.f;
#endif
  kernel_x_0 = ic << 2;
  kernel_x_1 = kernel_x_0 + 1;
  kernel_x_2 = kernel_x_0 + 2;
  kernel_x_3 = kernel_x_0 + 3;
  int in_idx = mul24(ic, input_shape.x);
  for (int k_y = valid_kernel_height, idx_h = kernel_start_y; k_y >= 0;
       k_y -= stride_shape.y, idx_h++) {
    int in_y_idx = mad24(out_n_idx, input_shape.y, idx_h);
    int in_nh_value = select(in_y_idx, -1, idx_h < 0 || idx_h >= input_shape.y);
    int in_width0 = kernel_start_x;

    for (int k_x = valid_kernel_width; k_x >= 0; k_x -= stride_shape.x) {
      kernel_y = mad24(k_y, kernel_shape.x, k_x);

      weights0 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_0, kernel_y));
      weights1 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_1, kernel_y));
      weights2 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_2, kernel_y));
      weights3 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_3, kernel_y));

      int in_width_value0 = in_width0;
      in_width_value0 =
          select(in_idx + in_width_value0,
                 -1,
                 (in_width_value0 < 0 || in_width_value0 >= input_shape.x));
      in0 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_width_value0, in_nh_value));

      out0.x += in0.x * weights0.x;
      out0.y += in0.y * weights1.x;
      out0.z += in0.z * weights2.x;
      out0.w += in0.w * weights3.x;
      in_width0++;
    }
  }
  int2 out_pos0 =
      (int2)(out_c_blk_idx * output_shape.x + out_w_idx, out_nh_idx);
  out0 = activation_type4(out0, 0.f);
#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos0, out0);
}
