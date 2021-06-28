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
                   __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  const int pos_out_x = mad24(out_c, out_width, out_w);

#ifdef POOL_AVG

  CL_DTYPE4 res = (CL_DTYPE4)(0.0f);
  int div;
#ifdef EXCLUSIVE
  div = (end_h - start_h) * (end_w - start_w);
#else
  div = ksize_w * ksize_h;
#endif  // EXCLUSIVE

#ifdef GLOBAL
  // pool_avg_global: force to use fp32 to avoid the loss of accuracy
  float4 res_f32 = 0.f;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      res_f32 +=
          read_imagef(input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  res_f32 /= (float)div;
#ifdef CL_DTYPE_half
  res = convert_half4(res_f32);
#else
  res = res_f32;
#endif

#else
  // pool_avg: use default precision
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      res += READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  res /= (CL_DTYPE)div;
#endif  // GLOBAL

#else

  // POOL_MAX
  CL_DTYPE4 res = (CL_DTYPE4)(-FLT_MAX);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
      res = max(res, tmp);
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
                         __private const int local_block_size,
                         __private const int2 local_block_size_wh,
                         __private const int2 local_block_count_wh,
                         __local CL_DTYPE4* local_output) {
  const int out_c = get_global_id(0) / local_block_size;
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  // const int out_h = out_nh % out_height;
  const int out_h = out_nh - mul24(out_h, out_height);

  const int local_id = get_local_id(0);
  const int local_width_id = local_id % local_block_size_wh.x;
  const int local_height_id = local_id / local_block_size_wh.x;

  const int input_start = mul24(out_n, in_height);
  const int input_channel_start = mul24(out_c, in_width);
  const int input_height_start = mad24(out_h, stride_h, -pad_top);
  const int input_width_start = mad24(out_w, stride_w, -pad_left);

#ifdef POOL_AVG
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
    const int kernel_height_start = max(0, input_height_start);
    const int kernel_width_start = max(0, input_width_start);
    const int kernel_height_end = min(input_height_start + ksize_h, in_height);
    const int kernel_width_end = min(input_width_start + ksize_w, in_width);
#ifdef EXCLUSIVE
    const int block_size = mul24((kernel_height_end - kernel_height_start),
                                 (kernel_width_end - kernel_width_start));
#else
    const int block_size = ksize_w * ksize_h;
#endif  // EXCLUSIVE
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

__kernel void pool_max(__read_only image2d_t input,
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
                       __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  CL_DTYPE4 max_value = (CL_DTYPE4)(MIN_VALUE);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
      max_value = max(max_value, tmp);
    }
  }

  const int pos_out_x = mad24(out_c, out_width, out_w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), max_value);
}

__kernel void pool_avg(__read_only image2d_t input,
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
                       __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  CL_DTYPE4 sum = (CL_DTYPE4)(0.0f);

  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      sum += READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  CL_DTYPE div;
#ifdef EXCLUSIVE
  div = (CL_DTYPE)((end_h - start_h) * (end_w - start_w));
#else
  div = (CL_DTYPE)(ksize_w * ksize_h);
#endif
  CL_DTYPE4 avg = sum / div;
  const int pos_out_x = mad24(out_c, out_width, out_w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), avg);
}

__kernel void pool_avg_global(__read_only image2d_t input,
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
                              __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);   // =1
  const int out_nh = get_global_id(2);  // = n*1

  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  // do not use dtype4 here
  // skip issue for half 2048
  float4 sum = (float4)(0.0f);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  for (int y = 0; y < in_height; ++y) {
    for (int x = 0; x < in_width; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));

      sum = convert_float4(tmp) + sum;
    }
  }
  const float global_size_div = 1.0f / (in_height * in_width);
  CL_DTYPE4 avg;
  avg.x = CONVERT_TYPE_TO((sum.x * global_size_div), CL_COMPUTE_DTYPE);
  avg.y = CONVERT_TYPE_TO((sum.y * global_size_div), CL_COMPUTE_DTYPE);
  avg.z = CONVERT_TYPE_TO((sum.z * global_size_div), CL_COMPUTE_DTYPE);
  avg.w = CONVERT_TYPE_TO((sum.w * global_size_div), CL_COMPUTE_DTYPE);

#ifdef DEBUG
  if (out_c == 0) {
    printf("\033[31msum.x= %f \033 \n  ", sum.x);
    printf("sum.y=%f \n  ", sum.y);
    printf("sum.z=%f \n  ", sum.z);
    printf("sum.w=%f \n  ", sum.w);
    printf("one4.x=%f \n  ", convert_float(one4.x));

    printf("in_height=%d \n  ", in_height);
    printf("in_width=%d \n  ", in_width);
    printf("ksize_h=%d \n  ", ksize_h);
    printf("ksize_w=%d \n  ", ksize_w);
    printf("stride_h=%d \n  ", stride_h);
    printf("stride_w=%d \n  ", stride_w);
    printf("pad_top=%d \n  ", pad_top);
    printf("pad_left=%d \n  ", pad_left);
    printf("out_width=%d \n  ", out_width);
    printf("out_height=%d \n  ", out_height);
    printf("i++=%d \n  ", i++);
    printf("avg.x=%f \n  ", convert_float(avg.x));
    printf("avg.y=%f \n  ", convert_float(avg.y));
    printf("avg.z=%f \n  ", convert_float(avg.z));
    printf("avg.w=%f \n  ", convert_float(avg.w));
  }
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_nh), avg);
}
__kernel void pool_max_global(__read_only image2d_t input,
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
                              __private const int pad_left) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);   // =1
  const int out_nh = get_global_id(2);  // = n*1

  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  CL_DTYPE4 max_value = (CL_DTYPE4)(MIN_VALUE);
  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  for (int y = 0; y < in_height; ++y) {
    for (int x = 0; x < in_width; ++x) {
      max_value = max(max_value,
                      READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    input,
                                    SAMPLER,
                                    (int2)(pos_in_x + x, pos_in_y + y)));
    }
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_c, out_nh), max_value);
}
