/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
__kernel void softmax_width_buffer(__global const CL_DTYPE* input,
                                   __global CL_DTYPE* output,
                                   __private const int N,
                                   __private const int C,
                                   __private const int H,
                                   __private const int W,
                                   __private const float4 mask) {
  int c = get_global_id(0);
  int bh = get_global_id(1);

  const int b = bh / H;
  const int h = bh % H;
  const int offset = ((b * C + c) * H + h) * W + 0;

  if (c < C && bh < N * H) {
    /*Compute Max */
    CL_DTYPE4 max_value_v4 = vload4(0, input + offset);
    for (int i = 1; i < W; i += 4) {
      int tmpi = (i + 4 > W) ? W - 4 : i;
      max_value_v4 = fmax(max_value_v4, vload4(0, input + offset + tmpi));
    }
    CL_DTYPE max_value = max(max(max_value_v4.s0, max_value_v4.s1),
                             max(max_value_v4.s2, max_value_v4.s3));
    /*Compute Exp Sum*/
    float4 sum_value_v4 = (float4)0;
    for (int i = 0; i < W; i += 4) {
      int tmpi = (i + 4 > W) ? W - 4 : i;
      float4 maski = (i + 4 > W) ? mask : (float4)(1.0f);
      sum_value_v4 += exp(convert_float4(vload4(0, input + offset + tmpi)) -
                          (float4)max_value) *
                      maski;
    }
    float sum_value =
        sum_value_v4.s0 + sum_value_v4.s1 + sum_value_v4.s2 + sum_value_v4.s3;
    /*Compute Result */
    for (int i = 0; i < W; i += 4) {
      int tmpi = (i + 4 > W) ? W - 4 : i;
      CL_DTYPE4 value =
          CONVERT_TYPE_TO(convert_float4(exp(vload4(0, input + offset + tmpi) -
                                             (CL_DTYPE4)max_value)) /
                              (float4)sum_value,
                          CL_DTYPE4);
      vstore4(value, 0, output + offset + tmpi);
    }
  }
}

__kernel void softmax_height_buffer(__global const CL_DTYPE* input,
                                    __global CL_DTYPE* output,
                                    __private const int N,
                                    __private const int C,
                                    __private const int H,
                                    __private const int W) {
  int wc = get_global_id(0);
  int b = get_global_id(1);
  const int w_4 = (W + 3) / 4;
  const int c = wc / w_4;  // w4
  int w = (wc % w_4) << 2;
  w = (w + 4 > W) ? W - 4 : w;
  const int offset = (b * C + c) * H * W + w + 0 * W;
  if (wc < C * W && b < N) {
    /*Compute Max */
    CL_DTYPE4 max_value = vload4(0, input + offset);
    for (int i = 1; i < H; ++i) {
      max_value = max(max_value, vload4(0, input + offset + i * W));
    }
    /*Compute Exp Sum*/
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    for (int i = 0; i < H; ++i) {
      sum_value += exp(vload4(0, input + offset + i * W) - max_value);
    }
    /*Compute Result */
    for (int i = 0; i < H; ++i) {
      CL_DTYPE4 value =
          exp(vload4(0, input + offset + i * W) - max_value) / sum_value;
      vstore4(value, 0, output + offset + i * W);
    }
  }
}

__kernel void softmax_channel_buffer(__global const CL_DTYPE* input,
                                     __global CL_DTYPE* output,
                                     __private const int N,
                                     __private const int C,
                                     __private const int H,
                                     __private const int W) {
  int hw = get_global_id(0);
  int b = get_global_id(1);
  const int w_4 = (W + 3) / 4;
  const int h = hw / w_4;  // w4
  int w = (hw % w_4) << 2;
  w = (w + 4 > W) ? W - 4 : w;
  const int offset = b * C * H * W + h * W + w;
  const int ch_dim = H * W;
  if (hw < H * W && b < N) {
    /*Compute Max */
    CL_DTYPE4 max_value = vload4(0, input + offset);
    for (int i = 1; i < C; ++i) {
      max_value = max(max_value, vload4(0, input + offset + i * ch_dim));
    }
    /*Compute Exp Sum*/
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    for (int i = 0; i < C; ++i) {
      sum_value += exp(vload4(0, input + offset + i * ch_dim) - max_value);
    }
    /*Compute Result */
    for (int i = 0; i < C; ++i) {
      CL_DTYPE4 value =
          exp(vload4(0, input + offset + i * ch_dim) - max_value) / sum_value;
      vstore4(value, 0, output + offset + i * ch_dim);
    }
  }
}

__kernel void softmax_batch_buffer(__global const CL_DTYPE* input,
                                   __global CL_DTYPE* output,
                                   __private const int N,
                                   __private const int C,
                                   __private const int H,
                                   __private const int W) {
  int hw = get_global_id(0);
  int c = get_global_id(1);
  const int w_4 = (W + 3) / 4;
  const int h = hw / w_4;  // w4
  int w = (hw % w_4) << 2;
  w = (w + 4 > W) ? W - 4 : w;
  const int offset = c * H * W + h * W + w;
  const int batch_dim = C * H * W;

  if (hw < H * W && c < C) {
    /*Compute Max */
    CL_DTYPE4 max_value = vload4(0, input + offset);
    for (int i = 1; i < N; ++i) {
      max_value = max(max_value, vload4(0, input + offset + i * batch_dim));
    }
    /*Compute Exp Sum*/
    CL_DTYPE4 sum_value = (CL_DTYPE4)(0.0f);
    for (int i = 0; i < N; ++i) {
      sum_value += exp(vload4(0, input + offset + i * batch_dim) - max_value);
    }
    /*Compute Result */
    for (int i = 0; i < N; ++i) {
      CL_DTYPE4 value =
          exp(vload4(0, input + offset + i * batch_dim) - max_value) /
          sum_value;
      vstore4(value, 0, output + offset + i * batch_dim);
    }
  }
}

__kernel void softmax_1x1_buffer(__global const CL_DTYPE* input,
                                 __global CL_DTYPE* output,
                                 __private const int c_count,
                                 __private const int c_blks) {
  const int c_blk_idx = get_global_id(0);
  const int b_idx = get_global_id(1);
  const int tid = get_local_id(0);

  int offset = b_idx * c_count;

  __local float4 tmp[8];
  __local float* tmpx1 = (__local float*)tmp;

  // Compute Max
  CL_DTYPE4 maxs = vload4(0, input + offset);
  for (int s = tid; s < c_blks; s += 32) {
    int tmpi = (s << 2);
    tmpi = (tmpi + 4 > c_count) ? c_count - 4 : tmpi;
    maxs = max(maxs, vload4(0, input + offset + tmpi));
  }
  maxs.x = max(maxs.x, maxs.y);
  maxs.z = max(maxs.z, maxs.w);
  maxs.x = max(maxs.x, maxs.z);
  tmpx1[tid] = (float)maxs.x;

  barrier(CLK_LOCAL_MEM_FENCE);

  float maximum;
  float4 maxx4;
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
  float sum = 0.f;
  for (int s = tid; s < c_blks; s += 32) {
    for (int i = 0; i < 4; i++) {
      int tmpi = (s << 2) + i;
      sum +=
          (tmpi < c_count) ? exp((float)input[offset + tmpi] - maximum) : 0.f;
    }
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
    int c_offset = (c_blk_idx << 2);
    c_offset = c_offset + 4 > c_count ? c_count - 4 : c_offset;
    CL_DTYPE4 src = vload4(0, input + offset + c_offset) - (CL_DTYPE4)maximum;
#ifdef CL_DTYPE_half
    CL_DTYPE4 res = convert_half4(exp(convert_float4(src)) * sum);
#else
    CL_DTYPE4 res = exp(src) * sum;
#endif
    vstore4(res, 0, output + offset + c_offset);
  }
}

__kernel void softmax_common_buffer(__global const CL_DTYPE* input,
                                    __global CL_DTYPE* output,
                                    __private const int pre_dim,
                                    __private const int select_range,
                                    __private const int select_dim) {
  int prefix = get_global_id(0);
  int suffix = get_global_id(1);

  int offset = prefix * pre_dim + suffix;

  /*Compute Exp Sum*/
  CL_DTYPE max_value = input[offset];
  for (int i = 1; i < select_range; i++) {
    max_value = max(max_value, input[offset + i * select_dim]);
  }

  /*Compute Exp Sum*/
  float sum_value = 0.0f;
  for (int i = 0; i < select_range; i++) {
    sum_value += exp((float)(input[offset + i * select_dim] - max_value));
  }

  /*Compute Result */
  for (int i = 0; i < select_range; i++) {
    CL_DTYPE value = CONVERT_TYPE_TO(
        exp((float)(input[offset + i * select_dim] - max_value)) /
            (float)sum_value,
        CL_DTYPE);
    output[offset + i * select_dim] = value;
  }
}