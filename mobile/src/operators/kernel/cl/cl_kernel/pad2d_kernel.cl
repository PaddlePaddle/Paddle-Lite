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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pad2d(
    __private const int in_height, __private const int in_width,
    __private const int out_height, __private const int out_width,
    __private const int pad_top, __private const int pad_bottom,
    __private const int pad_left, __private const int pad_right,
    __private const int mode, __private const float pad_value,
    __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int2 output_pos = (int2)(mad24(out_c, out_width, out_w), out_nh);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int x = out_w - pad_left;
  int y = out_h - pad_top;

  if (mode == 0) {
    if (x < 0 || y < 0 || x >= in_width || y >= in_height) {
      write_imageh(output, output_pos, (half4)(pad_value));
    } else {
      write_imageh(output, output_pos, read_imageh(input, sampler, (int2)(out_c * in_width + x, out_n * in_height + y)));
    }
  } else if (mode == 1) {
      x = abs(x);
      y = abs(y);
      x = x < in_width ? x : 2 * in_width - 2 - x;
      y = y < in_height ? y : 2 * in_height - 2 - y;
      write_imageh(output, output_pos, read_imageh(input, sampler, (int2)(out_c * in_width + x, out_n * in_height + y)));
  } else if (mode == 2) {
      x = x > 0 ? x : 0;
      x = x < in_width ? x : in_width - 1;
      y = y > 0 ? y : 0;
      y = y < in_height ? y : in_height - 1;
      write_imageh(output, output_pos, read_imageh(input, sampler, (int2)(out_c * in_width + x, out_n * in_height + y)));
  }
}
