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
                                   __private const int W) {
  int c = get_global_id(0);
  int bh = get_global_id(1);

  const int b = bh / H;
  const int h = bh % H;
  const int offset = ((b * C + c) * H + h) * W + 0;

  if (c < C && bh < N * H) {
    /*Compute Max */
    CL_DTYPE4 max_value_v4 = vload4(0, input + offset);
    for (int i = 1; i < W; i += 4) {
      max_value_v4 = fmax(max_value_v4, vload4(0, input + offset + i));
    }
    CL_DTYPE max_value = max(max(max_value_v4.s0, max_value_v4.s1),
                             max(max_value_v4.s2, max_value_v4.s3));
    /*Compute Exp Sum*/
    float4 sum_value_v4 = (float4)0;
    for (int i = 0; i < W; i += 4) {
      sum_value_v4 += exp(convert_float4(vload4(0, input + offset + i)) -
                          (float4)max_value);
    }
    float sum_value =
        sum_value_v4.s0 + sum_value_v4.s1 + sum_value_v4.s2 + sum_value_v4.s3;
    /*Compute Result */
    for (int i = 0; i < W; i += 4) {
      CL_DTYPE4 value =
          CONVERT_TYPE_TO(convert_float4(exp(vload4(0, input + offset + i) -
                                             (CL_DTYPE4)max_value)) /
                              (float4)sum_value,
                          CL_DTYPE4);
      vstore4(value, 0, output + offset + i);
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
  const int w = (wc % w_4) << 2;
  // const int offset = ((b * C + c) * H + 0) * W + w;
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
