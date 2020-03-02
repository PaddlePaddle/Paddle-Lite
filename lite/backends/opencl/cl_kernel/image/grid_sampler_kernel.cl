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
__kernel void buffer_to_image2d(__global CL_DTYPE *input,
                                __global CL_DTYPE *grid,
                                __write_only image2d_t output,
                                __private const int out_height,
                                __private const int out_width) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2) * 4;
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;
  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords1, coords2, outpoints;
  coords1.x = out_h / 4 * 2;
  coords1.y = out_n * out_width + out_w;
  coords2.x = coords1.x + 1;
  coords2.y = coords1.y;
  outpoints.x = out_c * out_width + out_w;
  outpoints.x = out_n * out_height + out_h;
  
  CL_DTYPE4 g1 = READ_IMG_TYPE(CL_DTYPE_CHAR, grid, sampler, coords1);
  CL_DTYPE4 g2 = READ_IMG_TYPE(CL_DTYPE_CHAR, grid, sampler, coords2);
  
  // x
  CL_DTYPE x = (g1.x + 1) * (out_width - 1) / 2;
  CL_DTYPE y = (g2.x + 1) * (out_height - 1) / 2;
  int x0 = floor(x);
  int y0 = floor(y);
  int x_p = out_c * out_width + x0;
  int y_p = out_n * out_height + y0;

  CL_DTYPE xs = x - x0;
  CL_DTYPE xe = x0 + 1 - x;
  CL_DTYPE ys = y - y0;
  CL_DTYPE ye = y0 + 1 - y;

  CL_DTYPE4 input0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p));
  CL_DTYPE4 input1 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p));
  CL_DTYPE4 input2 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p + 1));
  CL_DTYPE4 input3 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p + 1));

  CL_DTYPE4 out_val = input0 * xe * ye + input1 * xs * ye + input2 * xe * ys + input3 * xs * ys;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, outpoints, out_val);
 
  // y
  x = (g1.y + 1) * (out_width - 1) / 2;
  y = (g2.y + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;

  xs = x - x0;
  xe = x0 + 1 - x;
  ys = y - y0;
  ye = y0 + 1 - y;

  input0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p));
  input1 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p));
  input2 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p + 1));
  input3 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p + 1));

  out_val = input0 * xe * ye + input1 * xs * ye + input2 * xe * ys + input3 * xs * ys;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(outpoints.x, outpoints,y + 1), out_val);

  // z
  x = (g1.z + 1) * (out_width - 1) / 2;
  y = (g2.z + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;

  xs = x - x0;
  xe = x0 + 1 - x;
  ys = y - y0;
  ye = y0 + 1 - y;

  input0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p));
  input1 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p));
  input2 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p + 1));
  input3 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p + 1));

  out_val = input0 * xe * ye + input1 * xs * ye + input2 * xe * ys + input3 * xs * ys;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(outpoints.x, outpoints,y + 2), out_val);

  // w
  x = (g1.w + 1) * (out_width - 1) / 2;
  y = (g2.w + 1) * (out_height - 1) / 2;
  x0 = floor(x);
  y0 = floor(y);
  x_p = out_c * out_width + x0;
  y_p = out_n * out_height + y0;
  
  xs = x - x0;
  xe = x0 + 1 - x;
  ys = y - y0;
  ye = y0 + 1 - y;

  input0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p));
  input1 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p));
  input2 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p, y_p + 1));
  input3 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x_p + 1, y_p + 1));

  out_val = input0 * xe * ye + input1 * xs * ye + input2 * xe * ys + input3 * xs * ys;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(outpoints.x, outpoints,y + 3), out_val);
}