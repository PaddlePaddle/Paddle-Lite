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

__kernel void pad2d_constant(
    __read_only image2d_t input, __write_only image2d_t output,
    const int in_height, const int in_width,
    const int out_height, const int out_width,
    const int pad_h0, const int pad_h1,
    const int pad_w0, const int pad_w1,
    const float pad_value) {
        
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int2 output_pos = (int2)(mad24(out_c, out_width, out_w), out_nh);

  int x = out_w - pad_w0;
  int y = out_h - pad_h0;

  if (x < 0 || y < 0 || x >= in_width || y >= in_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, (CL_DTYPE4)(pad_value));
  } else {
    int2 coor = (int2)(out_c * in_width + x, out_n * in_height + y);
    CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coor);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, pixel);
  }
}

__kernel void pad2d_reflect(
    __read_only image2d_t input, __write_only image2d_t output,
    const int in_height, const int in_width,
    const int out_height, const int out_width,
    const int pad_h0, const int pad_h1,
    const int pad_w0, const int pad_w1,
    const float pad_value) {
        
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int2 output_pos = (int2)(mad24(out_c, out_width, out_w), out_nh);

  int x = out_w - pad_w0;
  int y = out_h - pad_h0;

  x = abs(x);
  y = abs(y);
  x = x < in_width ? x : 2 * in_width - 2 - x;
  y = y < in_height ? y : 2 * in_height - 2 - y;
  int2 coor = (int2)(out_c * in_width + x, out_n * in_height + y);
  CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coor);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, pixel);
}

__kernel void pad2d_edge(
    __read_only image2d_t input, __write_only image2d_t output,
    const int in_height, const int in_width,
    const int out_height, const int out_width,
    const int pad_h0, const int pad_h1,
    const int pad_w0, const int pad_w1,
    const float pad_value) {
        
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int2 output_pos = (int2)(mad24(out_c, out_width, out_w), out_nh);

  int x = out_w - pad_w0;
  int y = out_h - pad_h0;

  x = x > 0 ? x : 0;
  x = x < in_width ? x : in_width - 1;
  y = y > 0 ? y : 0;
  y = y < in_height ? y : in_height - 1;
  int2 coor = (int2)(out_c * in_width + x, out_n * in_height + y);
  CL_DTYPE4 pixel = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coor);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, pixel);
}
