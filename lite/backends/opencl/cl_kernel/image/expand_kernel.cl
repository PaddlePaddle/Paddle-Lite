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
__kernel void expand_cn0(__private const int OUT_C,
                         __private const int OUT_W,
                         __private const int OUT_NH,

                         __private const int IN_C,
                         __private const int IN_W,
                         __private const int IN_NH,

                         __private const int input_width,  /* of one block */
                         __private const int input_height, /* of one block */
                         __private const int output_width,
                         __private const int output_height,

                         __read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int in_c_len,
                         __private const int out_c_len) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }
  const int IN_N = IN_NH / input_height;
  const int OUT_N = OUT_NH / output_height;
  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = out_c % IN_C;
  const int in_w = out_w % input_width;
  const int in_h = out_h % input_height;
  const int in_n = out_n % IN_N;

  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_c * IN_W + in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}

__kernel void expand_cn1(__private const int OUT_C,
                         __private const int OUT_W,
                         __private const int OUT_NH,

                         __private const int IN_C,
                         __private const int IN_W,
                         __private const int IN_NH,

                         __private const int input_width,
                         __private const int input_height,
                         __private const int output_width,
                         __private const int output_height,

                         __read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int in_c_len,
                         __private const int out_c_len) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }
  const int IN_N = IN_NH / input_height;
  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_w = out_w % input_width;
  const int in_h = out_h % input_height;
  const int in_n = out_n % IN_N;
  const int in_nh = in_n * input_height + in_h;
  const int remain = (OUT_C << 2) - out_c_len;
  CL_DTYPE4 out = (CL_DTYPE4)(0);

  // out.x
  int out_c_remain = (out_c << 2) % in_c_len;
  int in_c = (out_c_remain >> 2);
  int in_c_remain = out_c_remain % 4;
  int2 input_pos = (int2)(in_c * IN_W + in_w, in_nh);
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);

  if (in_c_remain == 0) {
    out.x = in.x;
  } else if (in_c_remain == 1) {
    out.x = in.y;
  } else if (in_c_remain == 2) {
    out.x = in.z;
  } else {
    out.x = in.w;
  }

  // out.y
  out_c_remain = ((out_c << 2) + 1) % in_c_len;
  in_c = (out_c_remain >> 2);
  in_c_remain = out_c_remain % 4;
  input_pos = (int2)(in_c * IN_W + in_w, in_nh);
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);

  if (in_c_remain == 0) {
    out.y = in.x;
  } else if (in_c_remain == 1) {
    out.y = in.y;
  } else if (in_c_remain == 2) {
    out.y = in.z;
  } else {
    out.y = in.w;
  }

  // out.z
  out_c_remain = ((out_c << 2) + 2) % in_c_len;
  in_c = (out_c_remain >> 2);
  in_c_remain = out_c_remain % 4;
  input_pos = (int2)(in_c * IN_W + in_w, in_nh);
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);

  if (in_c_remain == 0) {
    out.z = in.x;
  } else if (in_c_remain == 1) {
    out.z = in.y;
  } else if (in_c_remain == 2) {
    out.z = in.z;
  } else {
    out.z = in.w;
  }

  // out.w
  out_c_remain = ((out_c << 2) + 3) % in_c_len;
  in_c = (out_c_remain >> 2);
  in_c_remain = out_c_remain % 4;
  input_pos = (int2)(in_c * IN_W + in_w, in_nh);
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);

  if (in_c_remain == 0) {
    out.w = in.x;
  } else if (in_c_remain == 1) {
    out.w = in.y;
  } else if (in_c_remain == 2) {
    out.w = in.z;
  } else {
    out.w = in.w;
  }

  if (out_c == OUT_C - 1) {
    if (remain == 1) {
      out.w = 0;
    } else if (remain == 2) {
      out.z = 0;
      out.w = 0;
    } else if (remain == 3) {
      out.y = 0;
      out.z = 0;
      out.w = 0;
    }
  }
  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, out);
}