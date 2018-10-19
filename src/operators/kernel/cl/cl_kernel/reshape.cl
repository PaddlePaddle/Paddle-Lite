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

__kernel void reshape(__read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const int d0,
                      __private const int d1,
                      __private const int d2,
                      __private const int d3,
                      __private const int x0,
                      __private const int x1,
                      __private const int x2,
                      __private const int x3) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(x, y));

  write_imageh(output, (int2)(x, y), in);
}


/*

__kernel void reshape(__read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const int d0,
                      __private const int d1,
                      __private const int d2,
                      __private const int d3,
                      __private const int x0,
                      __private const int x1,
                      __private const int x2,
                      __private const int x3) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  int obx = x / x3;
  int oby = y / x2;
  int ox = x % x3;
  int oy = y % x2;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 r;
  for (int i = 0; i < 4; i++) {
    int t = obx * 4 + i;
    if (t > x1) break;
    int oindex = oby * x1 * x2 * x3 + t * x2 * x3 + ox * x3 + oy;
    int i3 = oindex % d3; oindex /= d3;
    int i2 = oindex % d2; oindex /= d2;
    int i1 = oindex % d1; oindex /= d1;
    int i0 = oindex;
    int ix = (i1 / 4) * d3 + i3;
    int iy = i0 * d2 + i2;
    half4 p = read_imageh(input, sampler, (int2)(ix, iy));
    ((half*)&r)[i] = ((half*)&p)[i1%4];
  }
  write_imageh(output, (int2)(x, y), r);
}

*/
