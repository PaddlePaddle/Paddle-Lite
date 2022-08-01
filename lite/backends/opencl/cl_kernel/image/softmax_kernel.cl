/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
__kernel void softmax_width(__read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int N,
                            __private const int C,
                            __private const int H,
                            __private const int W) {
  int c = get_global_id(0);
  int bh = get_global_id(1);
  if (c < C && bh < N * H) {
    // Compute Max
    CL_DTYPE4 max_value =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c * W, bh));
    for (int i = 1; i < W; ++i) {
      max_value = max(
          max_value,
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c * W + i, bh)));
    }
    // Compute Exp Sum
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    for (int i = 0; i < W; ++i) {
      sum_value += exp(
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c * W + i, bh)) -
          max_value);
    }
    // Compute Result
    for (int i = 0; i < W; ++i) {
      CL_DTYPE4 value =
          exp(READ_IMG_TYPE(
                  CL_DTYPE_CHAR, input, SAMPLER, (int2)(c * W + i, bh)) -
              max_value) /
          sum_value;
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(c * W + i, bh), value);
    }
  }
}
__kernel void softmax_height(__read_only image2d_t input,
                             __write_only image2d_t output,
                             __private const int N,
                             __private const int C,
                             __private const int H,
                             __private const int W) {
  int wc = get_global_id(0);
  int b = get_global_id(1);
  if (wc < C * W && b < N) {
    /*Compute Max */
    CL_DTYPE4 max_value =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, b * H));
    for (int i = 1; i < H; ++i) {
      max_value = max(
          max_value,
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, b * H + i)));
    }
    /*Compute Exp Sum*/
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    ;
    for (int i = 0; i < H; ++i) {
      sum_value += exp(
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, b * H + i)) -
          max_value);
    }
    /*Compute Result */
    for (int i = 0; i < H; ++i) {
      CL_DTYPE4 value =
          exp(READ_IMG_TYPE(
                  CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, b * H + i)) -
              max_value) /
          sum_value;
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(wc, b * H + i), value);
    }
  }
}

__kernel void softmax_batch(__read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int N,
                            __private const int C,
                            __private const int H,
                            __private const int W) {
  int wc = get_global_id(0);
  int h = get_global_id(1);

  if (wc < C * W && h < H) {
    // Compute Max
    CL_DTYPE4 max_value =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, h));
    for (int i = 1; i < N; ++i) {
      max_value = max(
          max_value,
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, i * H + h)));
    }
    // Compute Exp Sum
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    for (int i = 0; i < N; ++i) {
      sum_value += exp(
          READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, i * H + h)) -
          max_value);
    }
    // Compute Result
    for (int i = 0; i < N; ++i) {
      CL_DTYPE4 value =
          exp(READ_IMG_TYPE(
                  CL_DTYPE_CHAR, input, SAMPLER, (int2)(wc, i * H + h)) -
              max_value) /
          sum_value;
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(wc, i * H + h), value);
    }
  }
}

__kernel void softmax_channel(__read_only image2d_t input,
                              __write_only image2d_t output,
                              __private const int output_channels,
                              __private const int remain_channels,
                              __private const int global_size_dim0,
                              __private const int global_size_dim1) {
  const int c_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int bh_idx = get_global_id(2);

  CL_DTYPE4 input_data;
  CL_DTYPE max_value = (CL_DTYPE)(MIN_VALUE);
  for (int i = 0; i < global_size_dim0 - 1; ++i) {
    input_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(width_idx + i * global_size_dim1, bh_idx));
    max_value = max(max_value, input_data.x);
    max_value = max(max_value, input_data.y);
    max_value = max(max_value, input_data.z);
    max_value = max(max_value, input_data.w);
  }

  input_data = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      input,
      SAMPLER,
      (int2)(width_idx + (global_size_dim0 - 1) * global_size_dim1, bh_idx));
  if (remain_channels == 0) {
    max_value = max(max_value, input_data.w);
    max_value = max(max_value, input_data.z);
    max_value = max(max_value, input_data.y);
    max_value = max(max_value, input_data.x);
  } else if (remain_channels == 1) {
    max_value = max(max_value, input_data.z);
    max_value = max(max_value, input_data.y);
    max_value = max(max_value, input_data.x);
  } else if (remain_channels == 2) {
    max_value = max(max_value, input_data.y);
    max_value = max(max_value, input_data.x);
  } else if (remain_channels == 3) {
    max_value = max(max_value, input_data.x);
  }

  CL_DTYPE sum_value = (CL_DTYPE)(0.0f);
  for (short i = 0; i < global_size_dim0 - 1; ++i) {
    input_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(width_idx + i * global_size_dim1, bh_idx));
    input_data = exp(input_data - max_value);
    sum_value += input_data.x;
    sum_value += input_data.y;
    sum_value += input_data.z;
    sum_value += input_data.w;
  }

  input_data = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      input,
      SAMPLER,
      (int2)(width_idx + (global_size_dim0 - 1) * global_size_dim1, bh_idx));
  input_data -= max_value;
  if (remain_channels == 0) {
    sum_value += exp(input_data.w);
    sum_value += exp(input_data.z);
    sum_value += exp(input_data.y);
    sum_value += exp(input_data.x);
  } else if (remain_channels == 1) {
    sum_value += exp(input_data.z);
    sum_value += exp(input_data.y);
    sum_value += exp(input_data.x);
  } else if (remain_channels == 2) {
    sum_value += exp(input_data.y);
    sum_value += exp(input_data.x);
  } else if (remain_channels == 3) {
    sum_value += exp(input_data.x);
  }

  int cur_out_width_pos = mad24(c_blk_idx, global_size_dim1, width_idx);
  input_data =
      READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(cur_out_width_pos, bh_idx)) -
      max_value;
  const int output_remain = output_channels - mul24(c_blk_idx, 4);

  if (output_remain == 1) {
    input_data.x = exp(input_data.x) / sum_value;
  } else if (output_remain == 2) {
    input_data.y = exp(input_data.y) / sum_value;
    input_data.x = exp(input_data.x) / sum_value;
  } else if (output_remain == 3) {
    input_data.z = exp(input_data.z) / sum_value;
    input_data.y = exp(input_data.y) / sum_value;
    input_data.x = exp(input_data.x) / sum_value;
  } else {
    input_data = exp(input_data) / sum_value;
  }
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output, (int2)(cur_out_width_pos, bh_idx), input_data);
}

__kernel void softmax_1x1(__read_only image2d_t input,
                          __write_only image2d_t output,
                          __private const float4 mask,
                          __private const int c_blks) {
  const int c_blk_idx = get_global_id(0);
  const int b_idx = get_global_id(1);
  const int tid = get_local_id(0);

  // Compute Max
  float4 maxx4 = read_imagef(input, SAMPLER, (int2)(0, b_idx));
  for (int s = tid; s < c_blks; s += 32) {  // as workgroup size is 32
    float4 mask_a = s == c_blks - 1 ? mask : (float4)(1.0f);
    float4 src = read_imagef(input, SAMPLER, (int2)(s, b_idx));
    src = src * mask_a;
    maxx4 = max(maxx4, src);
  }
  float maximum = max(maxx4.x, maxx4.y);
  maximum = max(maximum, maxx4.z);
  maximum = max(maximum, maxx4.w);

  // We need to find the final max value among each workgroup.
  // Note workgroup size is 32, so we need 8 float4 data to store 32 maximum.
  __local float4 tmp[8];
  __local float* tmpx1 = (__local float*)tmp;
  tmpx1[tid] = maximum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    maxx4 = max(tmp[0], tmp[1]);
    maxx4 = max(maxx4, tmp[2]);
    maxx4 = max(maxx4, tmp[3]);
    maxx4 = max(maxx4, tmp[4]);
    maxx4 = max(maxx4, tmp[5]);
    maxx4 = max(maxx4, tmp[6]);
    maxx4 = max(maxx4, tmp[7]);
    maximum = max(maxx4.x, maxx4.y);
    maximum = max(maximum, maxx4.z);
    maximum = max(maximum, maxx4.w);
    tmpx1[0] = maximum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  maximum = tmpx1[0];

  // Compute Exp Sum
  float sum = 0.0f;
  for (int s = tid; s < c_blks; s += 32) {
    float4 mask_temp = s == c_blks - 1 ? mask : (float4)(1.0f);
    float4 src =
        read_imagef(input, SAMPLER, (int2)(s, b_idx)) - (float4)(maximum);
    sum += dot(mask_temp, exp(src));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = 1.0f / sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];

  // Compute Result
  if (c_blk_idx < c_blks) {
    float4 src = read_imagef(input, SAMPLER, (int2)(c_blk_idx, b_idx)) -
                 (float4)(maximum);
#ifdef CL_DTYPE_half
    CL_DTYPE4 res = convert_half4(exp(src) * sum);
#else
    CL_DTYPE4 res = exp(src) * sum;
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(c_blk_idx, b_idx), res);
  }
}
