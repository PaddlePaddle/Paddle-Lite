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

#include "cl_common.h"

__kernel void grid_sampler(__private const int out_height,
                           __private const int out_width,
                           __read_only image2d_t input,
                           __read_only image2d_t grid,
                           __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2) * 4;
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;
  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int x_grid = out_h / 4 * 2;
  int y_grid = out_n * out_width + out_w;
  float4 g1 = read_imagef(grid, sampler, (int2)(x_grid, y_grid));
  float4 g2 = read_imagef(grid, sampler, (int2)(x_grid + 1, y_grid));

  float x = (g1.x + 1) * (out_width - 1) / 2;
  float y = (g2.x + 1) * (out_height - 1) / 2;
  float x0 = floor(x);
  float y0 = floor(y);
  int x_p = out_c * out_width + x0;
  int y_p = out_n * out_height + y0;
  int x_out = out_c * out_width + out_w;
  int y_out = out_n * out_height + out_h;
  float4 input0 = read_imagef(input, sampler, (int2)(x_p,     y_p));
  float4 input1 = read_imagef(input, sampler, (int2)(x_p + 1, y_p));
  float4 input2 = read_imagef(input, sampler, (int2)(x_p,     y_p + 1));
  float4 input3 = read_imagef(input, sampler, (int2)(x_p + 1, y_p + 1));
  float4 out_val = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                                      input1 * (x - x0) * (y0 + 1 - y) +
                                      input2 * (x0 + 1 - x) * (y - y0) +
                                      input3 * (x - x0) * (y - y0);
  write_imageh(output, (int2)(x_out, y_out), convert_half4(out_val));

  x = (g1.y + 1) * (out_width - 1) / 2;
  y = (g2.y + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;
  input0 = read_imagef(input, sampler, (int2)(x_p,     y_p));
  input1 = read_imagef(input, sampler, (int2)(x_p + 1, y_p));
  input2 = read_imagef(input, sampler, (int2)(x_p,     y_p + 1));
  input3 = read_imagef(input, sampler, (int2)(x_p + 1, y_p + 1));
  out_val = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                                      input1 * (x - x0) * (y0 + 1 - y) +
                                      input2 * (x0 + 1 - x) * (y - y0) +
                                      input3 * (x - x0) * (y - y0);
  write_imageh(output, (int2)(x_out, y_out + 1), convert_half4(out_val));

  x = (g1.z + 1) * (out_width - 1) / 2;
  y = (g2.z + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;
  input0 = read_imagef(input, sampler, (int2)(x_p,     y_p));
  input1 = read_imagef(input, sampler, (int2)(x_p + 1, y_p));
  input2 = read_imagef(input, sampler, (int2)(x_p,     y_p + 1));
  input3 = read_imagef(input, sampler, (int2)(x_p + 1, y_p + 1));
  out_val = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                                      input1 * (x - x0) * (y0 + 1 - y) +
                                      input2 * (x0 + 1 - x) * (y - y0) +
                                      input3 * (x - x0) * (y - y0);
  write_imageh(output, (int2)(x_out, y_out + 2), convert_half4(out_val));

  x = (g1.w + 1) * (out_width - 1) / 2;
  y = (g2.w + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;
  input0 = read_imagef(input, sampler, (int2)(x_p,     y_p));
  input1 = read_imagef(input, sampler, (int2)(x_p + 1, y_p));
  input2 = read_imagef(input, sampler, (int2)(x_p,     y_p + 1));
  input3 = read_imagef(input, sampler, (int2)(x_p + 1, y_p + 1));
  out_val = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                                      input1 * (x - x0) * (y0 + 1 - y) +
                                      input2 * (x0 + 1 - x) * (y - y0) +
                                      input3 * (x - x0) * (y - y0);
  write_imageh(output, (int2)(x_out, y_out + 3), convert_half4(out_val));
}
