/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

inline CL_DTYPE compute_lrn_output(__read_only image2d_t input,
                                   int start,
                                   int end,
                                   int out_w,
                                   int out_nh,
                                   int out_width,
                                   CL_DTYPE in,
                                   float k,
                                   float alpha,
                                   float beta) {
  CL_DTYPE square = CONVERT_TYPE_TO(0.0f, CL_DTYPE);
  for (int i = start; i <= end; i++) {
    int input_c = i / 4;
    int2 input_pos;
    input_pos.x = input_c * out_width + out_w;
    input_pos.y = out_nh;
    CL_DTYPE4 input_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    int num = i % 4;
    switch (num) {
      case 0:
        square += input_data.x * input_data.x;
        break;
      case 1:
        square += input_data.y * input_data.y;
        break;
      case 2:
        square += input_data.z * input_data.z;
        break;
      case 3:
        square += input_data.w * input_data.w;
        break;
    }
  }
  return in / (pow(k + alpha * square, beta));
}

__kernel void lrn(__read_only image2d_t input,
                  __write_only image2d_t output,
                  __private const int out_C,
                  __private const int out_W,
                  __private const int local_size,
                  __private const float k,
                  __private const float alpha,
                  __private const float beta) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const int out_c0 = out_c * 4;
  const int out_c1 = out_c0 + 1;
  const int out_c2 = out_c0 + 2;
  const int out_c3 = out_c0 + 3;

  int2 out_pos;
  out_pos.x = out_c * out_W + out_w;
  out_pos.y = out_nh;
  CL_DTYPE4 in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, out_pos);
  CL_DTYPE4 out_val;

  int pad = local_size / 2;
  int start = out_c0 - pad;
  int end = out_c0 + pad;
  start = start > 0 ? start : 0;
  end = end < out_C - 1 ? end : out_C - 1;
  out_val.x = compute_lrn_output(
      input, start, end, out_w, out_nh, out_W, in0.x, k, alpha, beta);

  if (out_c1 < out_C) {
    start = out_c1 - pad;
    end = out_c1 + pad;
    out_val.y = compute_lrn_output(
        input, start, end, out_w, out_nh, out_W, in0.y, k, alpha, beta);
  }
  if (out_c2 < out_C) {
    start = out_c2 - pad;
    end = out_c2 + pad;
    out_val.z = compute_lrn_output(
        input, start, end, out_w, out_nh, out_W, in0.z, k, alpha, beta);
  }
  if (out_c2 < out_C) {
    start = out_c3 - pad;
    end = out_c3 + pad;
    out_val.w = compute_lrn_output(
        input, start, end, out_w, out_nh, out_W, in0.w, k, alpha, beta);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, out_val);
}
