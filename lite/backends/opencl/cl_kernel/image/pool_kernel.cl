/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

__kernel void pool(__read_only image2d_t input,
                   __write_only image2d_t output,
                   __private const int in_height,
                   __private const int in_width,
                   __private const int out_height,
                   __private const int out_width,
                   __private const int ksize_h,
                   __private const int ksize_w,
                   __private const int stride_h,
                   __private const int stride_w,
                   __private const int pad_top,
                   __private const int pad_left,
                   __private const int exclusive,
                   __private const int adaptive) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h, start_w, end_h, end_w;
  int pool_size = 1;
  if (adaptive == 1) {
    start_h = (out_h * in_height) / out_height;
    start_w = (out_w * in_width) / out_width;
    end_h = ((out_h + 1) * in_height + (out_height - 1)) / out_height;
    end_w = ((out_w + 1) * in_width + (out_width - 1)) / out_width;
  } else {
    start_h = out_h * stride_h - pad_top;
    start_w = out_w * stride_w - pad_left;
    end_h = min(start_h + ksize_h, in_height + pad_top);
    end_w = min(start_w + ksize_w, in_width + pad_left);
    pool_size = (end_h - start_h) * (end_w - start_w);
    start_h = max(start_h, 0);
    start_w = max(start_w, 0);
    end_h = min(end_h, in_height);
    end_w = min(end_w, in_width);
  }

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  const int pos_out_x = mad24(out_c, out_width, out_w);

#ifdef POOL_AVG

  // force to use fp32 to avoid the loss of accuracy
  float4 res_fp32 = 0.f;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      res_fp32 +=
          read_imagef(input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }

  if (exclusive == 1 || adaptive == 1) {
    pool_size = (end_h - start_h) * (end_w - start_w);
  }

  res_fp32 = res_fp32 / (float)pool_size;
#ifdef CL_DTYPE_half
  CL_DTYPE4 res = convert_half4(res_fp32);
#else
  CL_DTYPE4 res = res_fp32;
#endif

#else

  // POOL_MAX
  CL_DTYPE4 res = (CL_DTYPE4)(-FLT_MAX);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
      res = fmax(res, tmp);
    }
  }

#endif  // POOL_AVG

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), res);
}

__kernel void pool_local(__read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int in_height,
                         __private const int in_width,
                         __private const int out_height,
                         __private const int out_width,
                         __private const int ksize_h,
                         __private const int ksize_w,
                         __private const int stride_h,
                         __private const int stride_w,
                         __private const int pad_top,
                         __private const int pad_left,
                         __private const int exclusive,
                         __private const int adaptive,
                         __private const int local_block_size,
                         __private const int2 local_block_size_wh,
                         __private const int2 local_block_count_wh,
                         __local CL_DTYPE4* local_output) {
  const int out_c = get_global_id(0) / local_block_size;
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  // const int out_h = out_nh % out_height;
  const int out_h = out_nh - mul24(out_n, out_height);

  const int local_id = get_local_id(0);
  const int local_width_id = local_id % local_block_size_wh.x;
  const int local_height_id = local_id / local_block_size_wh.x;

  const int input_start = mul24(out_n, in_height);
  const int input_channel_start = mul24(out_c, in_width);
  const int input_height_start = mad24(out_h, stride_h, -pad_top);
  const int input_width_start = mad24(out_w, stride_w, -pad_left);

#ifdef POOL_AVG
  // 1. Get data from global memroy to local memory
  __local float4* avg_output = (__local float4*)local_output;
  avg_output[local_id] = (float4)0;
  int pos_h = local_height_id;

  for (int local_h_block_id = 0; local_h_block_id < local_block_count_wh.y;
       local_h_block_id++) {
    if (pos_h >= ksize_h) break;
    int pos_w = local_width_id;
    int input_height_idx = input_height_start + pos_h;
    input_height_idx =
        select(input_start + input_height_idx,
               -1,
               (input_height_idx < 0 || input_height_idx >= in_height));
    for (int local_w_block_id = 0; local_w_block_id < local_block_count_wh.x;
         local_w_block_id++) {
      if (pos_w >= ksize_w) break;
      int input_width_idx = input_width_start + pos_w;
      input_width_idx =
          select(input_channel_start + input_width_idx,
                 -1,
                 (input_width_idx < 0 || input_width_idx >= in_width));
      float4 input_data = read_imagef(
          input, SAMPLER, (int2)(input_width_idx, input_height_idx));
      avg_output[local_id] += input_data;
      pos_w += local_block_size_wh.x;
    }
    pos_h += local_block_size_wh.y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2. Reduce in each workgroup
  for (int stride_h = (local_block_size_wh.y >> 1); stride_h > 0;
       stride_h >>= 1) {
    if (local_height_id < stride_h) {
      avg_output[local_id] +=
          avg_output[local_id + stride_h * local_block_size_wh.x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int stride_w = (local_block_size_wh.x >> 1); stride_w > 0;
       stride_w >>= 1) {
    if (local_height_id == 0 && local_width_id < stride_w) {
      avg_output[local_id] += avg_output[local_id + stride_w];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    int block_size;
    if (exclusive == 1 || adaptive == 1) {
      const int kernel_height_start = max(0, input_height_start);
      const int kernel_width_start = max(0, input_width_start);
      const int kernel_height_end =
          min(input_height_start + ksize_h, in_height);
      const int kernel_width_end = min(input_width_start + ksize_w, in_width);
      block_size = mul24((kernel_height_end - kernel_height_start),
                         (kernel_width_end - kernel_width_start));
    } else {
      block_size = ksize_w * ksize_h;
    }
    avg_output[local_id] = avg_output[local_id] / (float)block_size;

    const int output_channel_width_idx = mad24(out_c, out_width, out_w);
#ifdef CL_DTYPE_half
    CL_DTYPE4 res = convert_half4(avg_output[local_id]);
#else
    CL_DTYPE4 res = avg_output[local_id];
#endif
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_channel_width_idx, out_nh), res);
  }
#else
  // 1. Get data from global memroy to local memory
  local_output[local_id] = (CL_DTYPE4)(-FLT_MAX);
  int pos_h = local_height_id;

  for (int local_h_block_id = 0; local_h_block_id < local_block_count_wh.y;
       local_h_block_id++) {
    if (pos_h >= ksize_h) break;
    int pos_w = local_width_id;
    int input_height_idx = input_height_start + pos_h;
    input_height_idx =
        select(input_start + input_height_idx,
               -1,
               (input_height_idx < 0 || input_height_idx >= in_height));
    if (input_height_idx != -1) {
      for (int local_w_block_id = 0; local_w_block_id < local_block_count_wh.x;
           local_w_block_id++) {
        if (pos_w >= ksize_w) break;
        int input_width_idx = input_width_start + pos_w;
        input_width_idx =
            select(input_channel_start + input_width_idx,
                   -1,
                   (input_width_idx < 0 || input_width_idx >= in_width));

        if (input_width_idx != -1) {
          CL_DTYPE4 input_data =
              READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(input_width_idx, input_height_idx));
          local_output[local_id] = fmax(input_data, local_output[local_id]);
        }
        pos_w += local_block_size_wh.x;
      }
    }
    pos_h += local_block_size_wh.y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2. Reduce in each workgroup
  for (int stride_h = (local_block_size_wh.y >> 1); stride_h > 0;
       stride_h >>= 1) {
    if (local_height_id < stride_h) {
      local_output[local_id] =
          fmax(local_output[local_id + stride_h * local_block_size_wh.x],
               local_output[local_id]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int stride_w = (local_block_size_wh.x >> 1); stride_w > 0;
       stride_w >>= 1) {
    if (local_height_id == 0 && local_width_id < stride_w) {
      local_output[local_id] =
          fmax(local_output[local_id + stride_w], local_output[local_id]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    const int output_channel_width_idx = mad24(out_c, out_width, out_w);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output,
                   (int2)(output_channel_width_idx, out_nh),
                   local_output[local_id]);
  }
#endif  // POOL_AVG
}
